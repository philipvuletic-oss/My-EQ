//! egui editor: full-window response graph with draggable band nodes.

use crate::analyzer::{Analyzer, Resolution, SpectrumMode, SpectrumTap, COLUMNS, column_freq, FLOOR_DB};
use crate::eq::{
    self, BandSettings, FilterKind, Placement, ResponseView, Slope, MAX_FREQ, MAX_GAIN_DB, MAX_Q,
    MIN_FREQ, MIN_Q, NUM_BANDS,
};
use crate::{BandParams, MyEqParams};
use nih_plug::prelude::*;
use nih_plug_egui::egui::{
    self, epaint::Vertex, pos2, vec2, Align, Align2, Color32, Context, Id, Layout, Mesh, Pos2,
    Rect, Sense, Shape, Stroke, Ui, UiBuilder,
};
use nih_plug_egui::resizable_window::ResizableWindow;
use nih_plug_egui::EguiState;
use std::sync::atomic::Ordering;
use std::sync::Arc;

const BAR_HEIGHT: f32 = 44.0;
const PAD_Y: f32 = 16.0;
const FREQ_LABELS: [(f64, &str); 10] = [
    (20.0, "20"),
    (50.0, "50"),
    (100.0, "100"),
    (200.0, "200"),
    (500.0, "500"),
    (1000.0, "1k"),
    (2000.0, "2k"),
    (5000.0, "5k"),
    (10000.0, "10k"),
    (20000.0, "20k"),
];

pub struct GuiState {
    selected: Option<usize>,
    analyzer: Analyzer,
    mode: SpectrumMode,
    resolution: Resolution,
}

impl GuiState {
    pub fn new(tap: Arc<SpectrumTap>) -> Self {
        Self {
            selected: None,
            analyzer: Analyzer::new(tap),
            mode: SpectrumMode::Both,
            resolution: Resolution::Medium,
        }
    }
}

pub fn band_color(index: usize) -> Color32 {
    let hue = (index as f32 * 0.381_966 + 0.02).fract();
    egui::ecolor::Hsva::new(hue, 0.62, 0.98, 1.0).into()
}

fn with_alpha(c: Color32, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

fn set<P: Param>(setter: &ParamSetter, param: &P, value: P::Plain) {
    setter.begin_set_parameter(param);
    setter.set_parameter(param, value);
    setter.end_set_parameter(param);
}

fn uses_gain(kind: FilterKind) -> bool {
    matches!(
        kind,
        FilterKind::Bell | FilterKind::LowShelf | FilterKind::HighShelf | FilterKind::TiltShelf
    )
}

fn uses_slope(kind: FilterKind) -> bool {
    matches!(kind, FilterKind::LowCut | FilterKind::HighCut)
}

/// Maps between plot coordinates and (frequency, gain).
#[derive(Clone, Copy)]
struct Plot {
    rect: Rect,
}

impl Plot {
    fn x_of(&self, freq: f64) -> f32 {
        let t = (freq.ln() - MIN_FREQ.ln()) / (MAX_FREQ.ln() - MIN_FREQ.ln());
        self.rect.left() + t as f32 * self.rect.width()
    }
    fn freq_at(&self, x: f32) -> f64 {
        let t = ((x - self.rect.left()) / self.rect.width()).clamp(0.0, 1.0) as f64;
        (MIN_FREQ.ln() + t * (MAX_FREQ.ln() - MIN_FREQ.ln())).exp()
    }
    fn y_of(&self, db: f64) -> f32 {
        let (top, h) = (self.rect.top() + PAD_Y, self.rect.height() - 2.0 * PAD_Y);
        top + ((MAX_GAIN_DB - db) / (2.0 * MAX_GAIN_DB)) as f32 * h
    }
    fn db_at(&self, y: f32) -> f64 {
        let (top, h) = (self.rect.top() + PAD_Y, self.rect.height() - 2.0 * PAD_Y);
        (MAX_GAIN_DB - ((y - top) / h) as f64 * 2.0 * MAX_GAIN_DB).clamp(-MAX_GAIN_DB, MAX_GAIN_DB)
    }
}

pub fn draw(
    ctx: &Context,
    setter: &ParamSetter,
    state: &mut GuiState,
    params: &Arc<MyEqParams>,
    egui_state: &EguiState,
    sample_rate: f64,
) {
    ResizableWindow::new("my_eq_resize")
        .min_size(vec2(640.0, 420.0))
        .show(ctx, egui_state, |ui| {
            let full = ui.max_rect();
            let plot = Plot {
                rect: Rect::from_min_max(full.min, pos2(full.right(), full.bottom() - BAR_HEIGHT)),
            };
            let bands = params.band_settings();

            draw_background(ui, full, plot);
            state.analyzer.set_resolution(state.resolution);
            state.analyzer.update(state.mode, sample_rate);
            if state.mode != SpectrumMode::Off {
                ui.ctx().request_repaint();
                draw_spectrum(ui, state, plot);
            }
            let bg = ui.interact(plot.rect, Id::new("graph_bg"), Sense::click_and_drag());
            draw_curves_and_nodes(ui, setter, state, params, &bands, plot, sample_rate);

            if bg.clicked() {
                state.selected = None;
            }
            if bg.double_clicked() {
                if let Some(pos) = bg.interact_pointer_pos() {
                    add_band(setter, state, params, &bands, plot, pos);
                }
            }

            draw_bottom_bar(ui, setter, state, params, full);
            if let Some(sel) = state.selected {
                draw_band_panel(ctx, setter, state, params, sel, full);
            }
        });
}

fn draw_background(ui: &Ui, full: Rect, plot: Plot) {
    let painter = ui.painter();
    let top = Color32::from_rgb(0x1b, 0x1e, 0x25);
    let bottom = Color32::from_rgb(0x10, 0x12, 0x16);
    let mut mesh = Mesh::default();
    let uv = egui::epaint::WHITE_UV;
    for (p, c) in [
        (full.left_top(), top),
        (full.right_top(), top),
        (full.right_bottom(), bottom),
        (full.left_bottom(), bottom),
    ] {
        mesh.vertices.push(Vertex { pos: p, uv, color: c });
    }
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    painter.add(Shape::mesh(mesh));

    let faint = Stroke::new(1.0, Color32::from_white_alpha(10));
    let zero = Stroke::new(1.0, Color32::from_white_alpha(28));
    let label_col = Color32::from_white_alpha(90);
    let font = egui::FontId::proportional(11.0);

    // Frequency grid: decades and their 2/3/4/5/... subdivisions.
    let mut decade = 10.0f64;
    while decade < MAX_FREQ {
        for m in 1..10 {
            let f = decade * m as f64;
            if !(MIN_FREQ..=MAX_FREQ).contains(&f) {
                continue;
            }
            let x = plot.x_of(f);
            let s = if m == 1 { Stroke::new(1.0, Color32::from_white_alpha(18)) } else { faint };
            painter.line_segment([pos2(x, plot.rect.top()), pos2(x, plot.rect.bottom())], s);
        }
        decade *= 10.0;
    }
    for (f, text) in FREQ_LABELS {
        painter.text(
            pos2(plot.x_of(f) + 3.0, plot.rect.bottom() - 4.0),
            Align2::LEFT_BOTTOM,
            text,
            font.clone(),
            label_col,
        );
    }
    // Gain grid.
    for db in (-30..=30).step_by(10) {
        let y = plot.y_of(db as f64);
        painter.line_segment(
            [pos2(plot.rect.left(), y), pos2(plot.rect.right(), y)],
            if db == 0 { zero } else { faint },
        );
        let text = if db > 0 { format!("+{db}") } else { format!("{db}") };
        painter.text(
            pos2(plot.rect.right() - 6.0, y - 2.0),
            Align2::RIGHT_BOTTOM,
            text,
            font.clone(),
            label_col,
        );
    }
}

const SPEC_DB_LO: f32 = -96.0;
const SPEC_DB_HI: f32 = 6.0;

fn draw_spectrum(ui: &mut Ui, state: &GuiState, plot: Plot) {
    let painter = ui.painter_at(plot.rect);
    let (top, bottom) = (plot.rect.top() + PAD_Y, plot.rect.bottom());
    let y_of = |db: f32| {
        let t = ((db - SPEC_DB_LO) / (SPEC_DB_HI - SPEC_DB_LO)).clamp(0.0, 1.0);
        bottom - t * (bottom - top)
    };
    let pts = |v: &[f32]| -> Vec<Pos2> {
        (0..COLUMNS)
            .map(|c| pos2(plot.x_of(column_freq(c as f32 + 0.5)), y_of(v[c])))
            .collect()
    };
    let a = &state.analyzer;
    if state.mode.post() {
        let p = pts(&a.post.smooth);
        let uv = egui::epaint::WHITE_UV;
        let mut mesh = Mesh::default();
        let (c_top, c_bot) = (Color32::from_rgba_unmultiplied(110, 135, 175, 90), Color32::from_rgba_unmultiplied(110, 135, 175, 8));
        for (k, pt) in p.iter().enumerate() {
            mesh.vertices.push(Vertex { pos: *pt, uv, color: c_top });
            mesh.vertices.push(Vertex { pos: pos2(pt.x, bottom), uv, color: c_bot });
            if k > 0 {
                let i = (k * 2) as u32;
                mesh.indices.extend_from_slice(&[i - 2, i - 1, i, i - 1, i + 1, i]);
            }
        }
        painter.add(Shape::mesh(mesh));
        painter.add(Shape::line(p, Stroke::new(1.0, Color32::from_rgba_unmultiplied(150, 175, 215, 140))));
    }
    if state.mode.pre() {
        painter.add(Shape::line(pts(&a.pre.smooth), Stroke::new(1.0, Color32::from_white_alpha(45))));
    }
    // Peak hold follows the primary (post if shown, otherwise pre) trace.
    let peak = if state.mode.post() { &a.post.peak } else { &a.pre.peak };
    let visible: Vec<Pos2> = pts(peak);
    if peak.iter().any(|&v| v > FLOOR_DB + 1.0) {
        painter.add(Shape::line(visible, Stroke::new(1.0, Color32::from_rgba_unmultiplied(200, 215, 240, 90))));
    }
}

fn view_for(p: Placement) -> ResponseView {
    match p {
        Placement::Right => ResponseView::Right,
        Placement::Mid => ResponseView::Mid,
        Placement::Side => ResponseView::Side,
        _ => ResponseView::Left,
    }
}

fn draw_curves_and_nodes(
    ui: &mut Ui,
    setter: &ParamSetter,
    state: &mut GuiState,
    params: &Arc<MyEqParams>,
    bands: &[BandSettings; NUM_BANDS],
    plot: Plot,
    sr: f64,
) {
    let n = ((plot.rect.width() / 2.0) as usize).clamp(64, 700);
    let freqs: Vec<f64> = (0..n)
        .map(|i| plot.freq_at(plot.rect.left() + plot.rect.width() * i as f32 / (n - 1) as f32))
        .collect();
    let zero_y = plot.y_of(0.0);
    let clamp_y = |y: f32| y.clamp(plot.rect.top(), plot.rect.bottom());
    let painter = ui.painter_at(plot.rect);
    let mut sum_db = vec![0.0f64; n];
    let mut any = false;

    for (i, b) in bands.iter().enumerate() {
        if !b.enabled {
            continue;
        }
        any = true;
        let color = band_color(i);
        let emphasis = ui.ctx().animate_bool_with_time(
            Id::new(("emph", i)),
            state.selected == Some(i) || ui.ctx().memory(|m| m.data.get_temp::<bool>(Id::new(("hov", i))).unwrap_or(false)),
            0.15,
        );
        let single = std::slice::from_ref(b);
        let mut own = Vec::with_capacity(n);
        for (k, &f) in freqs.iter().enumerate() {
            let left = eq::magnitude_response_db(single, sr, f, ResponseView::Left);
            sum_db[k] += left;
            own.push(if view_for(b.placement) == ResponseView::Left {
                left
            } else {
                eq::magnitude_response_db(single, sr, f, view_for(b.placement))
            });
        }
        let pts: Vec<Pos2> = own
            .iter()
            .enumerate()
            .map(|(k, &db)| {
                pos2(plot.x_of(freqs[k]), clamp_y(plot.y_of(db)))
            })
            .collect();
        let mut mesh = Mesh::default();
        let fill = with_alpha(color, (30.0 + 45.0 * emphasis) as u8);
        for (k, p) in pts.iter().enumerate() {
            let uv = egui::epaint::WHITE_UV;
            mesh.vertices.push(Vertex { pos: *p, uv, color: fill });
            mesh.vertices.push(Vertex { pos: pos2(p.x, zero_y), uv, color: fill });
            if k > 0 {
                let a = (k * 2) as u32;
                mesh.indices.extend_from_slice(&[a - 2, a - 1, a, a - 1, a + 1, a]);
            }
        }
        painter.add(Shape::mesh(mesh));
        painter.add(Shape::line(
            pts,
            Stroke::new(1.0 + emphasis, with_alpha(color, (150.0 + 100.0 * emphasis) as u8)),
        ));
    }

    // Summed response with a soft glow.
    let sum_pts: Vec<Pos2> = (0..n)
        .map(|k| pos2(plot.x_of(freqs[k]), clamp_y(plot.y_of(sum_db[k]))))
        .collect();
    if any {
        for (w, a) in [(9.0, 14u8), (5.0, 30)] {
            painter.add(Shape::line(sum_pts.clone(), Stroke::new(w, Color32::from_rgba_unmultiplied(150, 190, 255, a))));
        }
    }
    painter.add(Shape::line(sum_pts, Stroke::new(2.0, Color32::from_rgb(235, 242, 255))));

    // Nodes (drawn last so they sit above all curves).
    for i in 0..NUM_BANDS {
        let b = bands[i];
        let selected = state.selected == Some(i);
        if !b.enabled && !selected {
            continue;
        }
        node(ui, setter, state, params, i, b, plot);
    }
}

fn node(
    ui: &mut Ui,
    setter: &ParamSetter,
    state: &mut GuiState,
    params: &Arc<MyEqParams>,
    i: usize,
    b: BandSettings,
    plot: Plot,
) {
    let bp = &params.bands[i];
    let db = if uses_gain(b.kind) { b.gain_db } else { 0.0 };
    let center = pos2(plot.x_of(b.freq), plot.y_of(db));
    let resp = ui.interact(
        Rect::from_center_size(center, vec2(26.0, 26.0)),
        Id::new(("node", i)),
        Sense::click_and_drag(),
    );
    let hovered = resp.hovered() || resp.dragged();
    ui.ctx().memory_mut(|m| m.data.insert_temp(Id::new(("hov", i)), hovered));
    let anim = ui.ctx().animate_bool_with_time(Id::new(("nodehov", i)), hovered, 0.12);
    let selected = state.selected == Some(i);
    let shift = ui.input(|inp| inp.modifiers.shift);

    if resp.clicked() || resp.drag_started() || resp.secondary_clicked() {
        state.selected = Some(i);
    }
    if resp.dragged() {
        if let Some(pos) = resp.interact_pointer_pos() {
            let d = resp.drag_delta();
            let (nf, ng) = if shift {
                (
                    plot.freq_at(plot.x_of(b.freq) + d.x * 0.1),
                    (b.gain_db - (d.y * 0.1) as f64 * (2.0 * MAX_GAIN_DB) / plot.rect.height() as f64)
                        .clamp(-MAX_GAIN_DB, MAX_GAIN_DB),
                )
            } else {
                (plot.freq_at(pos.x), plot.db_at(pos.y))
            };
            set(setter, &bp.freq, nf.clamp(MIN_FREQ, MAX_FREQ) as f32);
            if uses_gain(b.kind) {
                set(setter, &bp.gain, ng as f32);
            }
        }
    }
    if resp.double_clicked() {
        set(setter, &bp.gain, 0.0);
    }
    if hovered {
        let scroll = ui.input(|inp| inp.raw_scroll_delta.y);
        if scroll != 0.0 {
            let k = if shift { 0.0004 } else { 0.002 };
            let q = (b.q * (scroll as f64 * k).exp()).clamp(MIN_Q, MAX_Q);
            set(setter, &bp.q, q as f32);
        }
    }
    resp.context_menu(|ui| node_menu(ui, setter, state, params, i));

    let color = band_color(i);
    let painter = ui.painter();
    let r = 6.0 + 2.5 * anim + if selected { 1.0 } else { 0.0 };
    if anim > 0.01 || selected {
        painter.circle_filled(center, r + 6.0 * anim.max(0.4), with_alpha(color, (40.0 + 40.0 * anim) as u8));
    }
    let fill = if b.enabled { color } else { Color32::from_gray(110) };
    painter.circle_filled(center, r, fill);
    painter.circle_stroke(center, r, Stroke::new(1.5, Color32::from_rgba_unmultiplied(15, 17, 22, 230)));
    if selected {
        painter.circle_stroke(center, r + 2.0, Stroke::new(1.5, Color32::WHITE));
    }
}

fn enum_menu<T: Enum + PartialEq + Copy + 'static>(
    ui: &mut Ui,
    title: &str,
    setter: &ParamSetter,
    param: &EnumParam<T>,
    enabled: bool,
) {
    ui.add_enabled_ui(enabled, |ui| {
        ui.menu_button(title, |ui| {
            let cur = param.value().to_index();
            for (idx, name) in T::variants().iter().enumerate() {
                if ui.selectable_label(cur == idx, *name).clicked() {
                    set(setter, param, T::from_index(idx));
                    ui.close_menu();
                }
            }
        });
    });
}

fn delete_band(setter: &ParamSetter, state: &mut GuiState, bp: &BandParams, i: usize) {
    set(setter, &bp.enabled, false);
    set(setter, &bp.gain, 0.0);
    if state.selected == Some(i) {
        state.selected = None;
    }
}

fn node_menu(ui: &mut Ui, setter: &ParamSetter, state: &mut GuiState, params: &MyEqParams, i: usize) {
    let bp = &params.bands[i];
    enum_menu(ui, "Type", setter, &bp.kind, true);
    enum_menu(ui, "Slope", setter, &bp.slope, uses_slope(bp.kind.value()));
    enum_menu(ui, "Stereo placement", setter, &bp.placement, true);
    ui.separator();
    if ui.button("Delete").clicked() {
        delete_band(setter, state, bp, i);
        ui.close_menu();
    }
}

fn add_band(
    setter: &ParamSetter,
    state: &mut GuiState,
    params: &MyEqParams,
    bands: &[BandSettings; NUM_BANDS],
    plot: Plot,
    pos: Pos2,
) {
    let Some(i) = (0..NUM_BANDS).find(|&i| !bands[i].enabled && state.selected != Some(i)) else {
        return;
    };
    let bp = &params.bands[i];
    set(setter, &bp.kind, FilterKind::Bell);
    set(setter, &bp.slope, Slope::Db12);
    set(setter, &bp.placement, Placement::Stereo);
    set(setter, &bp.freq, plot.freq_at(pos.x) as f32);
    set(setter, &bp.gain, 0.0);
    set(setter, &bp.q, 1.0);
    set(setter, &bp.enabled, true);
    state.selected = Some(i);
}

// ---------------------------------------------------------------------------
// Widgets
// ---------------------------------------------------------------------------

fn value_text<P: Param>(p: &P) -> String {
    p.normalized_value_to_string(p.unmodulated_normalized_value(), true)
}

/// Rotary knob bound to a parameter. Drag vertically; Shift for fine control;
/// double-click resets to the default.
fn knob<P: Param>(ui: &mut Ui, setter: &ParamSetter, param: &P, size: f32, color: Color32) {
    let (rect, resp) = ui.allocate_exact_size(vec2(size, size), Sense::click_and_drag());
    if resp.drag_started() {
        setter.begin_set_parameter(param);
    }
    if resp.dragged() {
        let fine = if ui.input(|i| i.modifiers.shift) { 0.1 } else { 1.0 };
        let n = param.unmodulated_normalized_value() - resp.drag_delta().y * 0.006 * fine;
        setter.set_parameter_normalized(param, n.clamp(0.0, 1.0));
    }
    if resp.drag_stopped() {
        setter.end_set_parameter(param);
    }
    if resp.double_clicked() {
        setter.begin_set_parameter(param);
        setter.set_parameter_normalized(param, param.default_normalized_value());
        setter.end_set_parameter(param);
    }
    let hover = ui.ctx().animate_bool_with_time(resp.id, resp.hovered() || resp.dragged(), 0.12);
    let value = param.unmodulated_normalized_value();
    let c = rect.center();
    let r = size * 0.5 - 3.0;
    let painter = ui.painter();
    let a0 = std::f32::consts::PI * 0.75;
    let sweep = std::f32::consts::PI * 1.5;
    let arc = |t: f32| -> Vec<Pos2> {
        let steps = ((t * 40.0) as usize).max(2);
        (0..=steps)
            .map(|s| {
                let a = a0 + sweep * t * s as f32 / steps as f32;
                c + vec2(a.cos(), a.sin()) * r
            })
            .collect()
    };
    painter.circle_filled(c, r - 3.0, Color32::from_rgb(36 + (10.0 * hover) as u8, 39 + (10.0 * hover) as u8, 47 + (10.0 * hover) as u8));
    painter.add(Shape::line(arc(1.0), Stroke::new(3.0, Color32::from_white_alpha(22))));
    if value > 0.001 {
        painter.add(Shape::line(arc(value), Stroke::new(3.0, color)));
    }
    let a = a0 + sweep * value;
    painter.line_segment(
        [c + vec2(a.cos(), a.sin()) * (r * 0.25), c + vec2(a.cos(), a.sin()) * (r - 5.0)],
        Stroke::new(2.0, Color32::WHITE),
    );
    resp.on_hover_text(format!("{}: {}", param.name(), value_text(param)));
}

fn labeled_knob<P: Param>(ui: &mut Ui, setter: &ParamSetter, param: &P, label: &str, color: Color32) {
    ui.vertical(|ui| {
        ui.set_width(62.0);
        ui.with_layout(Layout::top_down(Align::Center), |ui| {
            ui.label(egui::RichText::new(label).size(11.0).color(Color32::from_white_alpha(130)));
            knob(ui, setter, param, 42.0, color);
            ui.label(egui::RichText::new(value_text(param)).size(11.0).color(Color32::from_white_alpha(200)));
        });
    });
}

fn enum_combo<T: Enum + PartialEq + Copy + 'static>(
    ui: &mut Ui,
    id: Id,
    setter: &ParamSetter,
    param: &EnumParam<T>,
    enabled: bool,
) {
    let cur = param.value().to_index();
    ui.add_enabled_ui(enabled, |ui| {
        egui::ComboBox::from_id_salt(id)
            .width(104.0)
            .selected_text(T::variants()[cur])
            .show_ui(ui, |ui| {
                for (idx, name) in T::variants().iter().enumerate() {
                    if ui.selectable_label(cur == idx, *name).clicked() {
                        set(setter, param, T::from_index(idx));
                    }
                }
            });
    });
}

fn draw_band_panel(
    ctx: &Context,
    setter: &ParamSetter,
    state: &mut GuiState,
    params: &MyEqParams,
    i: usize,
    _full: Rect,
) {
    let bp = &params.bands[i];
    let color = band_color(i);
    let frame = egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(26, 29, 36, 238))
        .stroke(Stroke::new(1.0, Color32::from_white_alpha(28)))
        .corner_radius(14)
        .inner_margin(egui::Margin::same(12))
        .shadow(egui::Shadow { offset: [0, 4], blur: 16, spread: 0, color: Color32::from_black_alpha(120) });
    egui::Area::new(Id::new("band_panel"))
        .order(egui::Order::Foreground)
        .anchor(Align2::CENTER_BOTTOM, vec2(0.0, -(BAR_HEIGHT + 12.0)))
        .show(ctx, |ui| {
            frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 10.0;
                    labeled_knob(ui, setter, &bp.freq, "FREQ", color);
                    labeled_knob(ui, setter, &bp.gain, "GAIN", color);
                    labeled_knob(ui, setter, &bp.q, "Q", color);
                    ui.separator();
                    let kind = bp.kind.value();
                    egui::Grid::new("band_dropdowns").num_columns(2).show(ui, |ui| {
                        ui.label("Type");
                        enum_combo(ui, Id::new(("type", i)), setter, &bp.kind, true);
                        ui.end_row();
                        ui.label("Slope");
                        enum_combo(ui, Id::new(("slope", i)), setter, &bp.slope, uses_slope(kind));
                        ui.end_row();
                        ui.label("Stereo");
                        enum_combo(ui, Id::new(("place", i)), setter, &bp.placement, true);
                        ui.end_row();
                    });
                    ui.separator();
                    ui.vertical(|ui| {
                        let on = bp.enabled.value();
                        let text = if on { "On" } else { "Off" };
                        let btn = egui::Button::new(egui::RichText::new(text).strong())
                            .fill(if on { with_alpha(color, 90) } else { Color32::from_gray(50) })
                            .min_size(vec2(64.0, 26.0));
                        if ui.add(btn).clicked() {
                            set(setter, &bp.enabled, !on);
                        }
                        if ui.add(egui::Button::new("Delete").min_size(vec2(64.0, 26.0))).clicked() {
                            delete_band(setter, state, bp, i);
                        }
                    });
                });
            });
        });
}

fn draw_bottom_bar(ui: &mut Ui, setter: &ParamSetter, state: &mut GuiState, params: &MyEqParams, full: Rect) {
    let bar = Rect::from_min_max(pos2(full.left(), full.bottom() - BAR_HEIGHT), full.max);
    let painter = ui.painter();
    painter.rect_filled(bar, 0.0, Color32::from_rgb(0x12, 0x14, 0x19));
    painter.line_segment([bar.left_top(), bar.right_top()], Stroke::new(1.0, Color32::from_white_alpha(22)));
    let mut bar_ui = ui.new_child(
        UiBuilder::new()
            .max_rect(bar.shrink2(vec2(16.0, 4.0)))
            .layout(Layout::left_to_right(Align::Center)),
    );
    let dim = Color32::from_white_alpha(130);
    let accent = Color32::from_rgb(150, 190, 255);
    bar_ui.label(egui::RichText::new("MY EQ").strong().color(Color32::from_white_alpha(200)));
    bar_ui.add_space(24.0);
    bar_ui.label(egui::RichText::new("OUTPUT").size(11.0).color(dim));
    knob(&mut bar_ui, setter, &params.output_gain, 30.0, accent);
    bar_ui.label(egui::RichText::new(value_text(&params.output_gain)).size(11.0));
    bar_ui.add_space(16.0);
    bar_ui.label(egui::RichText::new("PAN").size(11.0).color(dim));
    knob(&mut bar_ui, setter, &params.output_pan, 30.0, accent);
    bar_ui.label(egui::RichText::new(value_text(&params.output_pan)).size(11.0));

    // Right-aligned analyzer controls.
    bar_ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
        for (r, name) in [(Resolution::High, "High"), (Resolution::Medium, "Med"), (Resolution::Low, "Low")] {
            if ui.selectable_label(state.resolution == r, name).clicked() {
                state.resolution = r;
            }
        }
        ui.label(egui::RichText::new("RES").size(11.0).color(dim));
        ui.add_space(16.0);
        for (m, name) in [
            (SpectrumMode::Both, "Pre+Post"),
            (SpectrumMode::Post, "Post"),
            (SpectrumMode::Pre, "Pre"),
            (SpectrumMode::Off, "Off"),
        ] {
            if ui.selectable_label(state.mode == m, name).clicked() {
                state.mode = m;
            }
        }
        ui.label(egui::RichText::new("ANALYZER").size(11.0).color(dim));
    });
}

pub fn sample_rate_of(sr: &nih_plug::prelude::AtomicF32) -> f64 {
    sr.load(Ordering::Relaxed) as f64
}
