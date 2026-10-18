use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, egui, widgets, EguiState};
use std::sync::Arc;

mod eq;

use eq::{
    BandSettings, FilterKind, ParametricEq, Placement, Slope, MAX_FREQ, MAX_GAIN_DB, MAX_Q,
    MIN_FREQ, MIN_Q, NUM_BANDS,
};

pub struct MyEq {
    params: Arc<MyEqParams>,
    eq: ParametricEq,
    sample_rate: f32,
}

#[derive(Params)]
struct BandParams {
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
        }
    }
}

#[derive(Params)]
struct MyEqParams {
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
    fn band_settings(&self) -> [BandSettings; NUM_BANDS] {
        std::array::from_fn(|i| self.bands[i].settings())
    }
}

impl Default for MyEq {
    fn default() -> Self {
        Self {
            params: Arc::new(MyEqParams::default()),
            eq: ParametricEq::new(),
            sample_rate: 44100.0,
        }
    }
}

impl Default for MyEqParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(300, 200),
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
        create_egui_editor(
            self.params.editor_state.clone(),
            (),
            |_, _| {},
            move |egui_ctx, setter, _state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    ui.heading("My EQ");
                    ui.label("Output Gain");
                    ui.add(widgets::ParamSlider::for_param(&params.output_gain, setter));
                    ui.label("Output Pan");
                    ui.add(widgets::ParamSlider::for_param(&params.output_pan, setter));
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;
        self.eq.set_sample_rate(buffer_config.sample_rate as f64);
        true
    }

    fn reset(&mut self) {
        self.eq.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let settings: [BandSettings; NUM_BANDS] = self.params.band_settings();
        self.eq.set_bands(&settings);

        for mut channel_samples in buffer.iter_samples() {
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
            let (l, r) = self.eq.process_sample(l, r);
            *channel_samples.get_mut(0).unwrap() = l * gl;
            *channel_samples.get_mut(1).unwrap() = r * gr;
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
