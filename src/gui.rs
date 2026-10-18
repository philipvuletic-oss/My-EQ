//! egui editor: full-window response graph with draggable band nodes.

use crate::analyzer::{Analyzer, Resolution, SpectrumMode, SpectrumTap, COLUMNS, column_freq, FLOOR_DB};
use crate::eq::{
    self, BandSettings, FilterKind, Placement, ResponseView, Slope, MAX_FREQ, MAX_GAIN_DB, MAX_Q,
    MIN_FREQ, MIN_Q, NUM_BANDS,
};
use crate::history::History;
use crate::levels::Levels;
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
const MIN_WIDTH: f32 = 900.0;
const RADIUS: u8 = 8;
const METER_W: f32 = 16.0;
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
    meters: Arc<eq::DynMeters>,
    levels: Arc<Levels>,
    history: History,
    styled: bool,
    meter_db: [f32; 2],
    meter_hold: [(f32, f64); 2],
}

impl GuiState {
    pub fn new(
        tap: Arc<SpectrumTap>,
        meters: Arc<eq::DynMeters>,
        levels: Arc<Levels>,
        params: &Arc<MyEqParams>,
    ) -> Self {
        Self {
            meters,
            levels,
            history: History::new(params.as_ref()),
            styled: false,
            meter_db: [-120.0; 2],
            meter_hold: [(-120.0, 0.0); 2],
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

pub(crate) fn set<P: Param>(setter: &ParamSetter, param: &P, value: P::Plain) {
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
    if !state.styled {
        apply_style(ctx);
        state.styled = true;
    }
    handle_history(ctx, setter, state);
    ResizableWindow::new("my_eq_resize")
        .min_size(vec2(MIN_WIDTH, 420.0))
        .show(ctx, egui_state, |ui| {
            let full = ui.max_rect();
            let plot = Plot {
                rect: Rect::from_min_max(full.min, pos2(full.right(), full.bottom() - BAR_HEIGHT)),
            };
            let mode = params.mode.value();
            let mut bands = params.band_settings();
            if mode == crate::ProcessingMode::LinearPhase {
                // Dynamics are inactive in Linear Phase mode; show static curves.
                bands.iter_mut().for_each(|b| b.dynamic = false);
            }
            let curve_sr = if mode == crate::ProcessingMode::NaturalPhase {
                sample_rate * 2.0
            } else {
                sample_rate
            };

            draw_background(ui, full, plot);
            state.analyzer.set_resolution(state.resolution);
            state.analyzer.update(state.mode, sample_rate);
            if state.mode != SpectrumMode::Off {
                ui.ctx().request_repaint();
                draw_spectrum(ui, state, plot);
            }
            let bg = ui.interact(plot.rect, Id::new("graph_bg"), Sense::click_and_drag());
            if bands.iter().any(|b| b.enabled && b.dynamic) {
                ui.ctx().request_repaint();
            }
            draw_curves_and_nodes(ui, setter, state, params, &bands, plot, curve_sr);

            if bg.clicked() {
                state.selected = None;
            }
            if bg.double_clicked() {
                if let Some(pos) = bg.interact_pointer_pos() {
                    add_band(setter, state, params, &bands, plot, pos);
                }
            }

            draw_meter(ui, state, plot);
            draw_bottom_bar(ui, setter, state, params, full);
            draw_toolbar(ctx, setter, state, params, sample_rate);
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
            pos2(plot.rect.right() - METER_W - 10.0, y - 2.0),
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

/// Band settings with the current dynamic gain offset applied.
fn live_settings(b: &BandSettings, offset: f64) -> BandSettings {
    let mut l = *b;
    if l.dynamic && l.kind.uses_gain() {
        l.gain_db = (l.gain_db + offset).clamp(-MAX_GAIN_DB, MAX_GAIN_DB);
    }
    l
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
        let live = live_settings(b, state.meters.get(i));
        let b = &live;
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
            // Band gains are shown after the global gain scale; undo it for the raw param.
            let scale = params.gain_scale.value() as f64;
            if uses_gain(b.kind) && scale.abs() >= 0.05 {
                set(setter, &bp.gain, (ng / scale).clamp(-MAX_GAIN_DB, MAX_GAIN_DB) as f32);
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
    if resp.dragged() {
        drag_tooltip(ui, center, &b, plot);
    }

    let color = band_color(i);
    let painter = ui.painter();
    if b.enabled && b.dynamic && uses_gain(b.kind) {
        // Translucent dynamic range, plus a marker at the live gain.
        let y0 = plot.y_of(b.gain_db);
        let y1 = plot.y_of((b.gain_db + b.range_db).clamp(-MAX_GAIN_DB, MAX_GAIN_DB));
        let band = Rect::from_x_y_ranges(center.x - 8.0..=center.x + 8.0, y0.min(y1)..=y0.max(y1));
        painter.rect_filled(band, 8.0, with_alpha(color, 55));
        painter.rect_stroke(band, 8.0, Stroke::new(1.0, with_alpha(color, 110)), egui::StrokeKind::Inside);
        let live_y = plot.y_of(live_settings(&b, state.meters.get(i)).gain_db);
        painter.circle_filled(pos2(center.x, live_y), 3.5, Color32::WHITE);
    }
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

fn format_freq(f: f64) -> String {
    if f >= 1000.0 {
        format!("{:.2} kHz", f / 1000.0)
    } else {
        format!("{:.1} Hz", f)
    }
}

/// Small rounded readout that follows the node while it is being dragged.
fn drag_tooltip(ui: &Ui, center: Pos2, b: &BandSettings, plot: Plot) {
    let text = if uses_gain(b.kind) {
        format!("{}   {:+.1} dB", format_freq(b.freq), b.gain_db)
    } else {
        format_freq(b.freq)
    };
    let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Tooltip, Id::new("drag_tip")));
    let galley = painter.layout_no_wrap(text, egui::FontId::proportional(12.0), Color32::WHITE);
    let size = galley.size() + vec2(16.0, 10.0);
    let mut min = center + vec2(14.0, -size.y - 12.0);
    min.x = min.x.min(plot.rect.right() - size.x - METER_W - 4.0).max(plot.rect.left() + 4.0);
    min.y = min.y.max(plot.rect.top() + 4.0);
    let rect = Rect::from_min_size(min, size);
    painter.add(egui::Shadow { offset: [0, 2], blur: 8, spread: 0, color: Color32::from_black_alpha(90) }.as_shape(rect, RADIUS));
    painter.rect_filled(rect, RADIUS, Color32::from_rgba_unmultiplied(30, 33, 41, 240));
    painter.rect_stroke(rect, RADIUS, Stroke::new(1.0, Color32::from_white_alpha(30)), egui::StrokeKind::Inside);
    painter.galley(rect.min + vec2(8.0, 5.0), galley, Color32::WHITE);
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
                ui.vertical(|ui| {
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
                ui.separator();
                ui.add_enabled_ui(uses_gain(bp.kind.value()), |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 10.0;
                        let dynamic = bp.dynamic.value();
                        let btn = egui::Button::new(
                            egui::RichText::new(if dynamic { "Dynamic: On" } else { "Dynamic: Off" }).strong(),
                        )
                        .fill(if dynamic { with_alpha(color, 90) } else { Color32::from_gray(50) })
                        .min_size(vec2(96.0, 26.0));
                        if ui.add(btn).clicked() {
                            set(setter, &bp.dynamic, !dynamic);
                        }
                        if dynamic {
                            labeled_knob(ui, setter, &bp.dyn_threshold, "THRESH", color);
                            labeled_knob(ui, setter, &bp.dyn_range, "RANGE", color);
                            labeled_knob(ui, setter, &bp.dyn_attack, "ATTACK", color);
                            labeled_knob(ui, setter, &bp.dyn_release, "RELEASE", color);
                            let sc = bp.dyn_sidechain.value();
                            let btn = egui::Button::new(if sc { "Sidechain: Ext" } else { "Sidechain: Int" })
                                .fill(if sc { with_alpha(color, 90) } else { Color32::from_gray(50) })
                                .min_size(vec2(104.0, 26.0));
                            if ui.add(btn).clicked() {
                                set(setter, &bp.dyn_sidechain, !sc);
                            }
                        }
                    });
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

    bar_ui.add_space(16.0);
    bar_ui.label(egui::RichText::new("GAIN SCALE").size(11.0).color(dim));
    knob(&mut bar_ui, setter, &params.gain_scale, 30.0, accent);
    bar_ui.label(egui::RichText::new(value_text(&params.gain_scale)).size(11.0));
    bar_ui.add_space(16.0);
    let auto = params.auto_gain.value();
    if bar_ui
        .selectable_label(auto, "Auto Gain")
        .on_hover_text("Match output loudness to input loudness")
        .clicked()
    {
        set(setter, &params.auto_gain, !auto);
    }
    if auto {
        bar_ui.label(egui::RichText::new(format!("{:+.1} dB", state.levels.auto_gain_db())).size(11.0));
    }

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


// ---------------------------------------------------------------------------
// Style, undo/redo, A/B, presets, meter
// ---------------------------------------------------------------------------

fn apply_style(ctx: &Context) {
    ctx.style_mut(|style| {
        let v = &mut style.visuals;
        let r = egui::CornerRadius::same(RADIUS);
        v.widgets.noninteractive.corner_radius = r;
        v.widgets.inactive.corner_radius = r;
        v.widgets.hovered.corner_radius = r;
        v.widgets.active.corner_radius = r;
        v.widgets.open.corner_radius = r;
        v.menu_corner_radius = r;
        v.window_corner_radius = egui::CornerRadius::same(RADIUS + 4);
        v.popup_shadow = egui::Shadow { offset: [0, 4], blur: 14, spread: 0, color: Color32::from_black_alpha(110) };
        v.widgets.inactive.weak_bg_fill = Color32::from_rgb(40, 44, 54);
        v.widgets.hovered.weak_bg_fill = Color32::from_rgb(54, 59, 72);
        style.animation_time = 0.12;
    });
}

fn handle_history(ctx: &Context, setter: &ParamSetter, state: &mut GuiState) {
    let (now, down) = ctx.input(|i| (i.time, i.pointer.any_down()));
    state.history.update(now, down);
    let (undo, redo) = ctx.input_mut(|i| {
        let redo = i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z)
            || i.consume_key(egui::Modifiers::COMMAND, egui::Key::Y);
        let undo = i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z);
        (undo, redo)
    });
    if redo {
        state.history.redo(setter);
    } else if undo {
        state.history.undo(setter);
    }
}

fn draw_toolbar(ctx: &Context, setter: &ParamSetter, state: &mut GuiState, params: &Arc<MyEqParams>, sample_rate: f64) {
    let frame = egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(26, 29, 36, 225))
        .stroke(Stroke::new(1.0, Color32::from_white_alpha(24)))
        .corner_radius(RADIUS + 2)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .shadow(egui::Shadow { offset: [0, 2], blur: 10, spread: 0, color: Color32::from_black_alpha(80) });
    egui::Area::new(Id::new("toolbar"))
        .order(egui::Order::Foreground)
        .anchor(Align2::LEFT_TOP, vec2(12.0, 12.0))
        .show(ctx, |ui| {
            frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    ui.menu_button("Presets  \u{25be}", |ui| {
                        for preset in PRESETS {
                            if ui.button(preset.name).clicked() {
                                apply_preset(setter, params, preset);
                                state.selected = None;
                                ui.close_menu();
                            }
                        }
                    });
                    ui.separator();
                    for (idx, name) in ["A", "B"].into_iter().enumerate() {
                        let active = state.history.ab_active == idx;
                        let btn = egui::Button::new(egui::RichText::new(name).strong())
                            .fill(if active { Color32::from_rgb(70, 95, 150) } else { Color32::from_rgb(40, 44, 54) })
                            .min_size(vec2(26.0, 22.0));
                        if ui.add(btn).on_hover_text("A/B comparison").clicked() {
                            state.history.switch_ab(setter, idx);
                        }
                    }
                    ui.separator();
                    draw_mode_controls(ui, setter, params, sample_rate);
                    ui.separator();
                    let undo = ui.add_enabled(state.history.can_undo(), egui::Button::new("\u{21b6}").min_size(vec2(26.0, 22.0)));
                    if undo.on_hover_text("Undo (Ctrl+Z)").clicked() {
                        state.history.undo(setter);
                    }
                    let redo = ui.add_enabled(state.history.can_redo(), egui::Button::new("\u{21b7}").min_size(vec2(26.0, 22.0)));
                    if redo.on_hover_text("Redo (Ctrl+Shift+Z)").clicked() {
                        state.history.redo(setter);
                    }
                });
            });
        });
}

fn draw_mode_controls(ui: &mut egui::Ui, setter: &ParamSetter, params: &Arc<MyEqParams>, sample_rate: f64) {
    use crate::linear::LinearLatency;
    use crate::ProcessingMode;
    let cur = params.mode.value();
    for (m, name, tip) in [
        (ProcessingMode::ZeroLatency, "Zero", "Zero latency (minimum phase)"),
        (ProcessingMode::NaturalPhase, "Natural", "Natural phase: 2x oversampled, analog-like response near Nyquist"),
        (ProcessingMode::LinearPhase, "Linear", "Linear phase FIR (adds latency; dynamics and sidechain inactive)"),
    ] {
        let btn = egui::Button::new(name)
            .fill(if cur == m { Color32::from_rgb(70, 95, 150) } else { Color32::from_rgb(40, 44, 54) })
            .min_size(vec2(0.0, 22.0));
        if ui.add(btn).on_hover_text(tip).clicked() {
            setter.begin_set_parameter(&params.mode);
            setter.set_parameter(&params.mode, m);
            setter.end_set_parameter(&params.mode);
        }
    }
    if cur == ProcessingMode::LinearPhase {
        let q = params.lin_quality.value();
        for (v, name) in [(LinearLatency::Low, "Low"), (LinearLatency::Medium, "Med"), (LinearLatency::High, "High")] {
            let btn = egui::Button::new(name)
                .fill(if q == v { Color32::from_rgb(70, 95, 150) } else { Color32::from_rgb(40, 44, 54) })
                .min_size(vec2(0.0, 22.0));
            let ms = v.latency() as f64 / sample_rate.max(1.0) * 1000.0;
            if ui.add(btn).on_hover_text(format!("Latency \u{2248} {ms:.0} ms")).clicked() {
                setter.begin_set_parameter(&params.lin_quality);
                setter.set_parameter(&params.lin_quality, v);
                setter.end_set_parameter(&params.lin_quality);
            }
        }
    }
}

struct PresetBand {
    kind: FilterKind,
    freq: f32,
    gain: f32,
    q: f32,
    slope: Slope,
    /// (threshold dB, range dB, attack ms, release ms)
    dynamic: Option<(f32, f32, f32, f32)>,
}

const fn pb(kind: FilterKind, freq: f32, gain: f32, q: f32, slope: Slope) -> PresetBand {
    PresetBand { kind, freq, gain, q, slope, dynamic: None }
}

struct Preset {
    name: &'static str,
    bands: &'static [PresetBand],
}

const PRESETS: &[Preset] = &[
    Preset { name: "Init (flat)", bands: &[] },
    Preset {
        name: "Vocal clean-up",
        bands: &[
            pb(FilterKind::LowCut, 80.0, 0.0, 0.707, Slope::Db18),
            pb(FilterKind::Bell, 300.0, -3.0, 1.2, Slope::Db12),
            pb(FilterKind::Bell, 3000.0, 1.5, 0.9, Slope::Db12),
            PresetBand {
                kind: FilterKind::Bell,
                freq: 6500.0,
                gain: 0.0,
                q: 3.0,
                slope: Slope::Db12,
                dynamic: Some((-35.0, -6.0, 2.0, 80.0)),
            },
            pb(FilterKind::HighShelf, 10000.0, 2.0, 0.707, Slope::Db12),
        ],
    },
    Preset {
        name: "Kick tighten",
        bands: &[
            pb(FilterKind::LowCut, 30.0, 0.0, 0.707, Slope::Db24),
            pb(FilterKind::Bell, 60.0, 3.0, 1.2, Slope::Db12),
            pb(FilterKind::Bell, 300.0, -4.0, 1.5, Slope::Db12),
            pb(FilterKind::Bell, 3500.0, 2.5, 1.0, Slope::Db12),
            pb(FilterKind::HighCut, 14000.0, 0.0, 0.707, Slope::Db12),
        ],
    },
    Preset {
        name: "Master air",
        bands: &[
            pb(FilterKind::LowCut, 25.0, 0.0, 0.707, Slope::Db12),
            pb(FilterKind::Bell, 250.0, -1.0, 0.6, Slope::Db12),
            pb(FilterKind::Bell, 3000.0, 0.8, 0.5, Slope::Db12),
            pb(FilterKind::HighShelf, 12000.0, 2.5, 0.6, Slope::Db12),
        ],
    },
];

fn apply_preset(setter: &ParamSetter, params: &MyEqParams, preset: &Preset) {
    for (i, bp) in params.bands.iter().enumerate() {
        match preset.bands.get(i) {
            Some(b) => {
                set(setter, &bp.kind, b.kind);
                set(setter, &bp.freq, b.freq);
                set(setter, &bp.gain, b.gain);
                set(setter, &bp.q, b.q);
                set(setter, &bp.slope, b.slope);
                set(setter, &bp.placement, Placement::Stereo);
                set(setter, &bp.dyn_sidechain, false);
                set(setter, &bp.dynamic, b.dynamic.is_some());
                if let Some((thr, range, att, rel)) = b.dynamic {
                    set(setter, &bp.dyn_threshold, thr);
                    set(setter, &bp.dyn_range, range);
                    set(setter, &bp.dyn_attack, att);
                    set(setter, &bp.dyn_release, rel);
                }
                set(setter, &bp.enabled, true);
            }
            None => {
                if bp.enabled.value() {
                    set(setter, &bp.enabled, false);
                }
                if bp.dynamic.value() {
                    set(setter, &bp.dynamic, false);
                }
            }
        }
    }
}

const METER_FLOOR_DB: f32 = -60.0;

/// Output level meter on the right edge: smoothed bar plus a short peak hold.
fn draw_meter(ui: &mut Ui, state: &mut GuiState, plot: Plot) {
    let (now, dt) = ui.input(|i| (i.time, i.stable_dt.min(0.1)));
    let peaks = state.levels.take_peaks();
    for ch in 0..2 {
        let db = 20.0 * peaks[ch].max(1e-6).log10();
        state.meter_db[ch] = if db > state.meter_db[ch] { db } else { (state.meter_db[ch] - 40.0 * dt).max(db) };
        let (hold, t) = state.meter_hold[ch];
        if db >= hold {
            state.meter_hold[ch] = (db, now);
        } else if now - t > 1.2 {
            state.meter_hold[ch].0 = (hold - 20.0 * dt).max(db);
        }
    }
    ui.ctx().request_repaint();

    let area = Rect::from_min_max(
        pos2(plot.rect.right() - METER_W - 4.0, plot.rect.top() + PAD_Y),
        pos2(plot.rect.right() - 4.0, plot.rect.bottom() - PAD_Y),
    );
    let painter = ui.painter();
    let y_of = |db: f32| {
        let t = ((db - METER_FLOOR_DB) / -METER_FLOOR_DB).clamp(0.0, 1.0);
        area.bottom() - t * area.height()
    };
    let bar_w = (METER_W - 2.0) / 2.0;
    for ch in 0..2 {
        let x0 = area.left() + ch as f32 * (bar_w + 2.0);
        let bar = Rect::from_min_max(pos2(x0, area.top()), pos2(x0 + bar_w, area.bottom()));
        painter.rect_filled(bar, 2.0, Color32::from_white_alpha(14));
        let steps = 48;
        for k in 0..steps {
            let db_lo = METER_FLOOR_DB + (-METER_FLOOR_DB) * k as f32 / steps as f32;
            let db_hi = METER_FLOOR_DB + (-METER_FLOOR_DB) * (k + 1) as f32 / steps as f32;
            if db_lo >= state.meter_db[ch] {
                break;
            }
            let color = if db_lo > -6.0 {
                Color32::from_rgb(235, 80, 70)
            } else if db_lo > -18.0 {
                Color32::from_rgb(230, 200, 80)
            } else {
                Color32::from_rgb(90, 200, 140)
            };
            let top = y_of(db_hi.min(state.meter_db[ch]));
            painter.rect_filled(Rect::from_min_max(pos2(x0, top), pos2(x0 + bar_w, y_of(db_lo))), 0.0, color);
        }
        let hy = y_of(state.meter_hold[ch].0);
        if state.meter_hold[ch].0 > METER_FLOOR_DB {
            painter.line_segment([pos2(x0, hy), pos2(x0 + bar_w, hy)], Stroke::new(1.5, Color32::from_white_alpha(200)));
        }
    }
}
