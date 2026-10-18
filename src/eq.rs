//! 24-band parametric EQ built from Cytomic/Simper SVF sections.
//!
//! Everything here is allocation-free: all state lives in fixed-size arrays.

use nih_plug::prelude::Enum;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub const NUM_BANDS: usize = 24;
/// 96 dB/oct needs 16 poles = 8 second-order sections.
pub const MAX_SECTIONS: usize = 8;

pub const MIN_FREQ: f64 = 10.0;
pub const MAX_FREQ: f64 = 30_000.0;
pub const MIN_Q: f64 = 0.025;
pub const MAX_Q: f64 = 40.0;
pub const MAX_GAIN_DB: f64 = 30.0;

/// Width (dB) over which the dynamic gain fades in above the threshold.
const DYN_KNEE_DB: f64 = 6.0;
/// Dynamic gain coefficients are refreshed every this many samples.
const DYN_UPDATE_MASK: u32 = 3;

/// Live dynamic gain offsets (dB), written by the audio thread for the GUI.
pub struct DynMeters(pub [AtomicU32; NUM_BANDS]);

impl DynMeters {
    pub fn new() -> Arc<Self> {
        Arc::new(Self(std::array::from_fn(|_| AtomicU32::new(0f32.to_bits()))))
    }
    pub fn get(&self, band: usize) -> f64 {
        f32::from_bits(self.0[band].load(Ordering::Relaxed)) as f64
    }
}

/// Smoothing time constant for per-sample parameter smoothing.
const SMOOTH_MS: f64 = 15.0;
/// Time for a band to fade in/out when toggled.
const MIX_MS: f64 = 10.0;

#[derive(Enum, Clone, Copy, PartialEq, Eq, Debug)]
pub enum FilterKind {
    #[name = "Bell"]
    Bell,
    #[name = "Low Shelf"]
    LowShelf,
    #[name = "High Shelf"]
    HighShelf,
    #[name = "Low Cut"]
    LowCut,
    #[name = "High Cut"]
    HighCut,
    #[name = "Notch"]
    Notch,
    #[name = "Band Pass"]
    BandPass,
    #[name = "Tilt Shelf"]
    TiltShelf,
}

#[derive(Enum, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slope {
    #[name = "6 dB/oct"]
    Db6,
    #[name = "12 dB/oct"]
    Db12,
    #[name = "18 dB/oct"]
    Db18,
    #[name = "24 dB/oct"]
    Db24,
    #[name = "36 dB/oct"]
    Db36,
    #[name = "48 dB/oct"]
    Db48,
    #[name = "72 dB/oct"]
    Db72,
    #[name = "96 dB/oct"]
    Db96,
}

impl Slope {
    /// Filter order (number of poles).
    pub fn order(self) -> usize {
        match self {
            Slope::Db6 => 1,
            Slope::Db12 => 2,
            Slope::Db18 => 3,
            Slope::Db24 => 4,
            Slope::Db36 => 6,
            Slope::Db48 => 8,
            Slope::Db72 => 12,
            Slope::Db96 => 16,
        }
    }
}

#[derive(Enum, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Placement {
    #[name = "Stereo"]
    Stereo,
    #[name = "Left"]
    Left,
    #[name = "Right"]
    Right,
    #[name = "Mid"]
    Mid,
    #[name = "Side"]
    Side,
}

/// Which signal a response curve describes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResponseView {
    Left,
    Right,
    Mid,
    Side,
}

#[derive(Clone, Copy, Debug)]
pub struct BandSettings {
    pub enabled: bool,
    pub kind: FilterKind,
    pub freq: f64,
    pub gain_db: f64,
    pub q: f64,
    pub slope: Slope,
    pub placement: Placement,
    pub dynamic: bool,
    pub threshold_db: f64,
    /// How far the gain moves (dB) when the detector is well above threshold.
    pub range_db: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub sidechain: bool,
}

impl FilterKind {
    /// Whether the gain parameter has any effect for this filter type.
    pub fn uses_gain(self) -> bool {
        matches!(
            self,
            FilterKind::Bell | FilterKind::LowShelf | FilterKind::HighShelf | FilterKind::TiltShelf
        )
    }
}

impl Default for BandSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            kind: FilterKind::Bell,
            freq: 1000.0,
            gain_db: 0.0,
            q: 0.707,
            slope: Slope::Db12,
            placement: Placement::Stereo,
            dynamic: false,
            threshold_db: -30.0,
            range_db: -6.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            sidechain: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Coefficients
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Coeffs {
    first_order: bool,
    g: f64,
    k: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    m0: f64,
    m1: f64,
    m2: f64,
}

impl Coeffs {
    const IDENTITY: Coeffs = Coeffs {
        first_order: false,
        g: 0.0,
        k: 1.0,
        a1: 1.0,
        a2: 0.0,
        a3: 0.0,
        m0: 1.0,
        m1: 0.0,
        m2: 0.0,
    };

    fn svf(g: f64, k: f64, m0: f64, m1: f64, m2: f64) -> Self {
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        Self { first_order: false, g, k, a1, a2, a3, m0, m1, m2 }
    }

    /// First-order TPT section: out = m0 * x + m2 * lowpass(x).
    fn one_pole(g: f64, m0: f64, m2: f64) -> Self {
        Self { first_order: true, g, k: 0.0, a1: g / (1.0 + g), a2: 0.0, a3: 0.0, m0, m1: 0.0, m2 }
    }

    /// Complex response at normalised angle `theta = pi * f / sr`.
    fn response(&self, theta: f64) -> C {
        if self.g == 0.0 {
            return C::new(self.m0 + self.m2, 0.0); // identity
        }
        let s = C::new(0.0, theta.tan() / self.g);
        if self.first_order {
            let lp = C::ONE.div(s.add(C::ONE));
            C::new(self.m0, 0.0).add(lp.scale(self.m2))
        } else {
            let d = s.mul(s).add(s.scale(self.k)).add(C::ONE);
            let inv = C::ONE.div(d);
            let bp = s.mul(inv);
            C::new(self.m0, 0.0)
                .add(bp.scale(self.m1))
                .add(inv.scale(self.m2))
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Design {
    pub sections: [Coeffs; MAX_SECTIONS],
    pub count: usize,
}

impl Design {
    const EMPTY: Design = Design { sections: [Coeffs::IDENTITY; MAX_SECTIONS], count: 0 };

    fn push(&mut self, c: Coeffs) {
        if self.count < MAX_SECTIONS {
            self.sections[self.count] = c;
            self.count += 1;
        }
    }
}

/// Designs the SVF section(s) for a band. No allocation.
pub fn design(kind: FilterKind, freq: f64, gain_db: f64, q: f64, slope: Slope, sr: f64) -> Design {
    let freq = freq.clamp(MIN_FREQ, (sr * 0.49).max(MIN_FREQ + 1.0));
    let q = q.clamp(MIN_Q, MAX_Q);
    let gain_db = gain_db.clamp(-MAX_GAIN_DB, MAX_GAIN_DB);
    let tan = (PI * freq / sr).tan();
    let a = 10f64.powf(gain_db / 40.0);
    let mut d = Design::EMPTY;

    match kind {
        FilterKind::Bell => {
            let k = 1.0 / (q * a);
            d.push(Coeffs::svf(tan, k, 1.0, k * (a * a - 1.0), 0.0));
        }
        FilterKind::LowShelf => d.push(low_shelf(tan, q, a)),
        FilterKind::HighShelf => d.push(high_shelf(tan, q, a)),
        FilterKind::TiltShelf => {
            // +gain/2 below, -gain/2 above the centre frequency.
            let h = 10f64.powf(gain_db / 80.0);
            d.push(low_shelf(tan, 0.5f64.sqrt(), h));
            d.push(high_shelf(tan, 0.5f64.sqrt(), 1.0 / h));
        }
        FilterKind::Notch => {
            let k = 1.0 / q;
            d.push(Coeffs::svf(tan, k, 1.0, -k, 0.0));
        }
        FilterKind::BandPass => {
            let k = 1.0 / q;
            d.push(Coeffs::svf(tan, k, 0.0, k, 0.0));
        }
        FilterKind::LowCut | FilterKind::HighCut => {
            let high_pass = kind == FilterKind::LowCut;
            let n = slope.order();
            for i in 0..n / 2 {
                // Butterworth pole pairs.
                let sec_q = 1.0 / (2.0 * (PI * (2 * i + 1) as f64 / (2 * n) as f64).sin());
                let k = 1.0 / sec_q;
                d.push(if high_pass {
                    Coeffs::svf(tan, k, 1.0, -k, -1.0)
                } else {
                    Coeffs::svf(tan, k, 0.0, 0.0, 1.0)
                });
            }
            if n % 2 == 1 {
                d.push(if high_pass {
                    Coeffs::one_pole(tan, 1.0, -1.0)
                } else {
                    Coeffs::one_pole(tan, 0.0, 1.0)
                });
            }
        }
    }
    d
}

fn low_shelf(tan: f64, q: f64, a: f64) -> Coeffs {
    let g = tan / a.sqrt();
    let k = 1.0 / q;
    Coeffs::svf(g, k, 1.0, k * (a - 1.0), a * a - 1.0)
}

fn high_shelf(tan: f64, q: f64, a: f64) -> Coeffs {
    let g = tan * a.sqrt();
    let k = 1.0 / q;
    Coeffs::svf(g, k, a * a, k * (1.0 - a) * a, 1.0 - a * a)
}

// ---------------------------------------------------------------------------
// Minimal complex arithmetic (for the response function)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const ONE: C = C { re: 1.0, im: 0.0 };
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn scale(self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
    fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}

fn band_complex_response(s: &BandSettings, sr: f64, freq: f64) -> C {
    let d = design(s.kind, s.freq, s.gain_db, s.q, s.slope, sr);
    let theta = PI * freq.clamp(0.0, sr * 0.5 - 1e-3) / sr;
    let mut h = C::ONE;
    for c in &d.sections[..d.count] {
        h = h.mul(c.response(theta));
    }
    h
}

/// Combined magnitude response (linear gain) of all enabled bands at `freq` Hz
/// for the chosen view. Allocation-free; safe to call from the GUI thread.
pub fn magnitude_response(
    bands: &[BandSettings],
    sr: f64,
    freq: f64,
    view: ResponseView,
) -> f64 {
    // Per-view complex response, tracked as the (L,R) or (M,S) diagonal.
    let mut total = C::ONE;
    for s in bands.iter().filter(|s| s.enabled) {
        let h = band_complex_response(s, sr, freq);
        let one = C::ONE;
        let half = |x: C, y: C| x.add(y).scale(0.5);
        let eff = match (s.placement, view) {
            (Placement::Stereo, _) => h,
            (Placement::Left, ResponseView::Left) => h,
            (Placement::Right, ResponseView::Right) => h,
            (Placement::Left, ResponseView::Right) | (Placement::Right, ResponseView::Left) => one,
            (Placement::Left | Placement::Right, _) => half(h, one),
            (Placement::Mid, ResponseView::Mid) | (Placement::Side, ResponseView::Side) => h,
            (Placement::Mid, ResponseView::Side) | (Placement::Side, ResponseView::Mid) => one,
            (Placement::Mid, _) => half(h, one),
            (Placement::Side, _) => half(one, h),
        };
        total = total.mul(eff);
    }
    total.norm()
}

pub fn magnitude_response_db(bands: &[BandSettings], sr: f64, freq: f64, view: ResponseView) -> f64 {
    20.0 * magnitude_response(bands, sr, freq, view).max(1e-9).log10()
}

// ---------------------------------------------------------------------------
// Runtime processing
// ---------------------------------------------------------------------------

type SectionState = [f64; 2];
type ChainState = [SectionState; MAX_SECTIONS];

#[inline]
fn tick(c: &Coeffs, st: &mut SectionState, x: f64) -> f64 {
    if c.first_order {
        let v = (x - st[0]) * c.a1;
        let lp = v + st[0];
        st[0] = lp + v;
        c.m0 * x + c.m2 * lp
    } else {
        let v3 = x - st[1];
        let v1 = c.a1 * st[0] + c.a2 * v3;
        let v2 = st[1] + c.a2 * st[0] + c.a3 * v3;
        st[0] = 2.0 * v1 - st[0];
        st[1] = 2.0 * v2 - st[1];
        c.m0 * x + c.m1 * v1 + c.m2 * v2
    }
}

#[inline]
fn run_chain(d: &Design, st: &mut ChainState, mut x: f64) -> f64 {
    for i in 0..d.count {
        x = tick(&d.sections[i], &mut st[i], x);
    }
    x
}

struct Band {
    target: BandSettings,
    ln_freq: f64,
    gain_db: f64,
    ln_q: f64,
    design: Design,
    settled: bool,
    mix: f64,
    state: [ChainState; 2],
    detect: Design,
    detect_state: ChainState,
    env: f64,
    dyn_db: f64,
    att: f64,
    rel: f64,
    tick: u32,
}

impl Band {
    fn new() -> Self {
        let t = BandSettings::default();
        Self {
            ln_freq: t.freq.ln(),
            gain_db: t.gain_db,
            ln_q: t.q.ln(),
            target: t,
            design: Design::EMPTY,
            settled: false,
            mix: 0.0,
            state: [[[0.0; 2]; MAX_SECTIONS]; 2],
            detect: Design::EMPTY,
            detect_state: [[0.0; 2]; MAX_SECTIONS],
            env: 0.0,
            dyn_db: 0.0,
            att: 0.0,
            rel: 0.0,
            tick: 0,
        }
    }

    fn reset_state(&mut self) {
        self.state = [[[0.0; 2]; MAX_SECTIONS]; 2];
        self.detect_state = [[0.0; 2]; MAX_SECTIONS];
        self.env = 0.0;
    }

    fn active(&self) -> bool {
        self.target.enabled || self.mix > 0.0
    }

    fn set_target(&mut self, t: BandSettings, sr: f64) {
        let was_active = self.active();
        let structural = t.kind != self.target.kind
            || t.slope != self.target.slope
            || t.placement != self.target.placement;
        if t.placement != self.target.placement || !was_active {
            self.reset_state();
        }
        self.att = 1.0 - (-1.0 / (t.attack_ms.max(0.01) * 0.001 * sr)).exp();
        self.rel = 1.0 - (-1.0 / (t.release_ms.max(0.01) * 0.001 * sr)).exp();
        if !was_active && t.enabled {
            // Snap so the band fades in at its final settings.
            self.ln_freq = t.freq.ln();
            self.gain_db = t.gain_db;
            self.ln_q = t.q.ln();
            self.settled = false;
        }
        let changed = t.freq != self.target.freq
            || t.gain_db != self.target.gain_db
            || t.q != self.target.q;
        self.target = t;
        if changed || structural {
            self.settled = false;
        }
        if !self.settled && !was_active {
            self.recompute(sr);
        }
    }

    fn recompute_main(&mut self, sr: f64) {
        self.design = design(
            self.target.kind,
            self.ln_freq.exp(),
            self.gain_db + self.dyn_db,
            self.ln_q.exp(),
            self.target.slope,
            sr,
        );
    }

    fn recompute(&mut self, sr: f64) {
        self.recompute_main(sr);
        // Detection uses the band's own filter shape at its static gain.
        self.detect = design(
            self.target.kind,
            self.ln_freq.exp(),
            self.gain_db,
            self.ln_q.exp(),
            self.target.slope,
            sr,
        );
    }

    #[inline]
    fn advance(&mut self, sr: f64, smooth: f64, mix_step: f64) {
        let goal = if self.target.enabled { 1.0 } else { 0.0 };
        if self.mix < goal {
            self.mix = (self.mix + mix_step).min(goal);
        } else if self.mix > goal {
            self.mix = (self.mix - mix_step).max(goal);
        }
        if !self.target.dynamic && self.dyn_db != 0.0 {
            // Dynamics switched off: glide back to the static gain.
            self.dyn_db *= 1.0 - smooth;
            if self.dyn_db.abs() < 1e-4 {
                self.dyn_db = 0.0;
            }
            self.recompute_main(sr);
        }
        if self.settled {
            return;
        }
        let tf = self.target.freq.ln();
        let tq = self.target.q.ln();
        self.ln_freq += (tf - self.ln_freq) * smooth;
        self.ln_q += (tq - self.ln_q) * smooth;
        self.gain_db += (self.target.gain_db - self.gain_db) * smooth;
        if (tf - self.ln_freq).abs() < 1e-5
            && (tq - self.ln_q).abs() < 1e-5
            && (self.target.gain_db - self.gain_db).abs() < 1e-4
        {
            self.ln_freq = tf;
            self.ln_q = tq;
            self.gain_db = self.target.gain_db;
            self.settled = true;
        }
        self.recompute(sr);
    }

    #[inline]
    fn process(&mut self, l: f64, r: f64, sc: (f64, f64), sr: f64) -> (f64, f64) {
        if self.target.dynamic && self.target.kind.uses_gain() {
            let (a, b) = if self.target.sidechain { sc } else { (l, r) };
            let x = match self.target.placement {
                Placement::Left => a,
                Placement::Right => b,
                Placement::Side => (a - b) * 0.5,
                Placement::Stereo | Placement::Mid => (a + b) * 0.5,
            };
            let level = run_chain(&self.detect, &mut self.detect_state, x).abs();
            let c = if level > self.env { self.att } else { self.rel };
            self.env += (level - self.env) * c;
            let env_db = 20.0 * self.env.max(1e-6).log10();
            let t = ((env_db - self.target.threshold_db) / DYN_KNEE_DB).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            self.dyn_db = self.target.range_db.clamp(-MAX_GAIN_DB, MAX_GAIN_DB) * t;
            self.tick = self.tick.wrapping_add(1);
            if self.tick & DYN_UPDATE_MASK == 0 {
                self.recompute_main(sr);
            }
        }
        let d = &self.design;
        let [sa, sb] = &mut self.state;
        let (wl, wr) = match self.target.placement {
            Placement::Stereo => (run_chain(d, sa, l), run_chain(d, sb, r)),
            Placement::Left => (run_chain(d, sa, l), r),
            Placement::Right => (l, run_chain(d, sb, r)),
            Placement::Mid => {
                let (m, s) = ((l + r) * 0.5, (l - r) * 0.5);
                let m = run_chain(d, sa, m);
                (m + s, m - s)
            }
            Placement::Side => {
                let (m, s) = ((l + r) * 0.5, (l - r) * 0.5);
                let s = run_chain(d, sb, s);
                (m + s, m - s)
            }
        };
        (l + self.mix * (wl - l), r + self.mix * (wr - r))
    }
}

pub struct ParametricEq {
    bands: Vec<Band>,
    sr: f64,
    smooth: f64,
    mix_step: f64,
    meters: Arc<DynMeters>,
}

impl ParametricEq {
    /// Allocates; call from `initialize`, not from the audio thread.
    pub fn new() -> Self {
        Self::with_meters(DynMeters::new())
    }

    pub fn with_meters(meters: Arc<DynMeters>) -> Self {
        let mut eq = Self {
            bands: (0..NUM_BANDS).map(|_| Band::new()).collect(),
            sr: 44100.0,
            smooth: 0.0,
            mix_step: 0.0,
            meters,
        };
        eq.set_sample_rate(44100.0);
        eq
    }

    pub fn set_sample_rate(&mut self, sr: f64) {
        self.sr = sr;
        self.smooth = 1.0 - (-1.0 / (SMOOTH_MS * 0.001 * sr)).exp();
        self.mix_step = 1.0 / (MIX_MS * 0.001 * sr);
        for b in &mut self.bands {
            b.settled = false;
            b.recompute(sr);
        }
    }

    pub fn reset(&mut self) {
        for b in &mut self.bands {
            b.reset_state();
        }
    }

    /// Update targets for all bands (call once per buffer).
    pub fn set_bands(&mut self, settings: &[BandSettings; NUM_BANDS]) {
        for (b, s) in self.bands.iter_mut().zip(settings.iter()) {
            b.set_target(*s, self.sr);
        }
    }

    #[inline]
    pub fn process_sample(&mut self, l: f32, r: f32, sc: (f32, f32)) -> (f32, f32) {
        let (mut l, mut r) = (l as f64, r as f64);
        let sc = (sc.0 as f64, sc.1 as f64);
        let (sr, smooth, step) = (self.sr, self.smooth, self.mix_step);
        for (i, b) in self.bands.iter_mut().enumerate() {
            if !b.active() {
                continue;
            }
            b.advance(sr, smooth, step);
            (l, r) = b.process(l, r, sc, sr);
            if b.target.dynamic || b.dyn_db != 0.0 {
                self.meters.0[i].store((b.dyn_db as f32).to_bits(), Ordering::Relaxed);
            }
        }
        (l as f32, r as f32)
    }
}

impl Default for ParametricEq {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(amp: f32, sidechain_amp: Option<f32>) -> f64 {
        let sr = 48000.0;
        let meters = DynMeters::new();
        let mut eq = ParametricEq::with_meters(meters.clone());
        eq.set_sample_rate(sr);
        let mut b = [BandSettings::default(); NUM_BANDS];
        b[0] = BandSettings {
            enabled: true,
            freq: 1000.0,
            gain_db: 0.0,
            q: 1.0,
            dynamic: true,
            threshold_db: -30.0,
            range_db: -12.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            sidechain: sidechain_amp.is_some(),
            ..Default::default()
        };
        // A static +12 dB bell shape makes the detector sensitive at 1 kHz.
        b[0].gain_db = 12.0;
        eq.set_bands(&b);
        for n in 0..48000 {
            let ph = (std::f64::consts::TAU * 1000.0 * n as f64 / sr).sin() as f32;
            let sc = sidechain_amp.map_or(0.0, |a| a * ph);
            eq.process_sample(amp * ph, amp * ph, (sc, sc));
        }
        meters.get(0)
    }

    #[test]
    fn dynamic_gain_follows_level() {
        assert!(run(0.5, None) < -11.0, "loud should reach range");
        assert!(run(0.0005, None).abs() < 0.5, "quiet should stay put");
        assert!(run(0.0005, Some(0.5)) < -11.0, "sidechain should drive it");
        assert!(run(0.5, Some(0.0005)).abs() < 0.5, "main input ignored with sidechain");
    }
}
