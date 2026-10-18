use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, EguiState};
use std::sync::Arc;

mod analyzer;
mod eq;
mod gui;

use eq::{
    BandSettings, FilterKind, ParametricEq, Placement, Slope, MAX_FREQ, MAX_GAIN_DB, MAX_Q,
    MIN_FREQ, MIN_Q, NUM_BANDS,
};

pub struct MyEq {
    params: Arc<MyEqParams>,
    eq: ParametricEq,
    sample_rate: Arc<AtomicF32>,
    tap: Arc<analyzer::SpectrumTap>,
    meters: Arc<eq::DynMeters>,
}

#[derive(Params)]
pub(crate) struct BandParams {
    #[id = "on"]
    enabled: BoolParam,
    #[id = "type"]
    kind: EnumParam<FilterKind>,
    #[id = "freq"]
    freq: FloatParam,
    #[id = "gain"]
    gain: FloatParam,
    #[id = "q"]
    q: FloatParam,
    #[id = "slope"]
    slope: EnumParam<Slope>,
    #[id = "dyn"]
    dynamic: BoolParam,
    #[id = "dthr"]
    dyn_threshold: FloatParam,
    #[id = "drng"]
    dyn_range: FloatParam,
    #[id = "datk"]
    dyn_attack: FloatParam,
    #[id = "drel"]
    dyn_release: FloatParam,
    #[id = "dsc"]
    dyn_sidechain: BoolParam,
    #[id = "place"]
    placement: EnumParam<Placement>,
}

/// Skew factor so the normalised midpoint maps to the geometric mean (log-like).
fn log_skew(min: f32, max: f32) -> f32 {
    let geo = (min * max).sqrt();
    0.5f32.ln() / ((geo - min) / (max - min)).ln()
}

impl BandParams {
    fn new(index: usize) -> Self {
        // Spread default frequencies logarithmically across the spectrum.
        let t = index as f32 / (NUM_BANDS - 1) as f32;
        let default_freq = 20.0 * (20_000.0f32 / 20.0).powf(t);
        Self {
            enabled: BoolParam::new("Enabled", false),
            kind: EnumParam::new("Type", FilterKind::Bell),
            freq: FloatParam::new(
                "Frequency",
                default_freq,
                FloatRange::Skewed {
                    min: MIN_FREQ as f32,
                    max: MAX_FREQ as f32,
                    factor: log_skew(MIN_FREQ as f32, MAX_FREQ as f32),
                },
            )
            .with_unit(" Hz")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),
            gain: FloatParam::new(
                "Gain",
                0.0,
                FloatRange::Linear {
                    min: -(MAX_GAIN_DB as f32),
                    max: MAX_GAIN_DB as f32,
                },
            )
            .with_unit(" dB")
            .with_step_size(0.01),
            q: FloatParam::new(
                "Q",
                0.707,
                FloatRange::Skewed {
                    min: MIN_Q as f32,
                    max: MAX_Q as f32,
                    factor: log_skew(MIN_Q as f32, MAX_Q as f32),
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3)),
            slope: EnumParam::new("Slope", Slope::Db12),
            dynamic: BoolParam::new("Dynamic", false),
            dyn_threshold: FloatParam::new(
                "Threshold",
                -30.0,
                FloatRange::Linear { min: -60.0, max: 0.0 },
            )
            .with_unit(" dB")
            .with_step_size(0.1),
            dyn_range: FloatParam::new(
                "Range",
                -6.0,
                FloatRange::Linear {
                    min: -(MAX_GAIN_DB as f32),
                    max: MAX_GAIN_DB as f32,
                },
            )
            .with_unit(" dB")
            .with_step_size(0.1),
            dyn_attack: FloatParam::new(
                "Attack",
                10.0,
                FloatRange::Skewed { min: 0.1, max: 500.0, factor: log_skew(0.1, 500.0) },
            )
            .with_unit(" ms")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),
            dyn_release: FloatParam::new(
                "Release",
                100.0,
                FloatRange::Skewed { min: 5.0, max: 5000.0, factor: log_skew(5.0, 5000.0) },
            )
            .with_unit(" ms")
            .with_value_to_string(formatters::v2s_f32_rounded(0)),
            dyn_sidechain: BoolParam::new("Sidechain", false),
            placement: EnumParam::new("Placement", Placement::Stereo),
        }
    }

    fn settings(&self) -> BandSettings {
        BandSettings {
            enabled: self.enabled.value(),
            kind: self.kind.value(),
            freq: self.freq.value() as f64,
            gain_db: self.gain.value() as f64,
            q: self.q.value() as f64,
            slope: self.slope.value(),
            placement: self.placement.value(),
            dynamic: self.dynamic.value(),
            threshold_db: self.dyn_threshold.value() as f64,
            range_db: self.dyn_range.value() as f64,
            attack_ms: self.dyn_attack.value() as f64,
            release_ms: self.dyn_release.value() as f64,
            sidechain: self.dyn_sidechain.value(),
        }
    }
}

#[derive(Params)]
pub(crate) struct MyEqParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,

    #[nested(array, group = "Band")]
    bands: [BandParams; NUM_BANDS],

    #[id = "out_gain"]
    output_gain: FloatParam,

    #[id = "out_pan"]
    output_pan: FloatParam,
}

impl MyEqParams {
    /// Snapshot of all band settings, e.g. for the GUI response curve via
    /// `eq::magnitude_response`.
    #[allow(dead_code)]
    pub(crate) fn band_settings(&self) -> [BandSettings; NUM_BANDS] {
        std::array::from_fn(|i| self.bands[i].settings())
    }
}

impl Default for MyEq {
    fn default() -> Self {
        let meters = eq::DynMeters::new();
        Self {
            params: Arc::new(MyEqParams::default()),
            eq: ParametricEq::with_meters(meters.clone()),
            meters,
            sample_rate: Arc::new(AtomicF32::new(44100.0)),
            tap: analyzer::SpectrumTap::new(),
        }
    }
}

impl Default for MyEqParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(1000, 560),
            bands: std::array::from_fn(BandParams::new),
            output_gain: FloatParam::new(
                "Output Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            output_pan: FloatParam::new(
                "Output Pan",
                0.0,
                FloatRange::Linear { min: -1.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),
        }
    }
}

impl Plugin for MyEq {
    const NAME: &'static str = "My EQ";
    const VENDOR: &'static str = "My EQ";
    const URL: &'static str = "https://github.com/philipvuletic-oss/My-EQ";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        aux_input_ports: &[new_nonzero_u32(2)],
        names: PortNames {
            aux_inputs: &["Sidechain"],
            ..PortNames::const_default()
        },
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let egui_state = self.params.editor_state.clone();
        let sample_rate = self.sample_rate.clone();
        let tap = self.tap.clone();
        let meters = self.meters.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            gui::GuiState::new(tap, meters),
            |_, _| {},
            move |egui_ctx, setter, state| {
                gui::draw(
                    egui_ctx,
                    setter,
                    state,
                    &params,
                    &egui_state,
                    gui::sample_rate_of(&sample_rate),
                );
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate.store(buffer_config.sample_rate, std::sync::atomic::Ordering::Relaxed);
        self.eq.set_sample_rate(buffer_config.sample_rate as f64);
        true
    }

    fn reset(&mut self) {
        self.eq.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let settings: [BandSettings; NUM_BANDS] = self.params.band_settings();
        self.eq.set_bands(&settings);

        let sidechain: &[&mut [f32]] = match aux.inputs.first_mut() {
            Some(b) if b.channels() >= 2 => b.as_slice_immutable(),
            _ => &[],
        };

        for (idx, mut channel_samples) in buffer.iter_samples().enumerate() {
            let sc = match sidechain {
                [l, r, ..] => (
                    l.get(idx).copied().unwrap_or(0.0),
                    r.get(idx).copied().unwrap_or(0.0),
                ),
                _ => (0.0, 0.0),
            };
            let gain = self.params.output_gain.smoothed.next();
            let pan = self.params.output_pan.smoothed.next();
            // Constant-power pan law, unity gain at centre.
            let angle = (pan + 1.0) * std::f32::consts::FRAC_PI_4;
            let (gl, gr) = (
                angle.cos() * std::f32::consts::SQRT_2 * gain,
                angle.sin() * std::f32::consts::SQRT_2 * gain,
            );
            let l = *channel_samples.get_mut(0).unwrap();
            let r = *channel_samples.get_mut(1).unwrap();
            self.tap.pre.push((l + r) * 0.5);
            let (l, r) = self.eq.process_sample(l, r, sc);
            let (l, r) = (l * gl, r * gr);
            self.tap.post.push((l + r) * 0.5);
            *channel_samples.get_mut(0).unwrap() = l;
            *channel_samples.get_mut(1).unwrap() = r;
        }
        ProcessStatus::Normal
    }
}

impl Vst3Plugin for MyEq {
    const VST3_CLASS_ID: [u8; 16] = *b"MyEqPluginGain01";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Eq];
}

nih_export_vst3!(MyEq);
