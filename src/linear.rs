//! Linear Phase mode: the EQ magnitude response is turned into a symmetric FIR that is
//! applied with uniformly partitioned FFT convolution (overlap-save).
//!
//! Kernel design allocates and is slow, so it runs on a background thread. The audio thread
//! only swaps `Box<Kernel>` pointers through try-locked slots and never allocates or frees.

use crate::eq::{self, BandSettings, Placement, NUM_BANDS};
use nih_plug::prelude::Enum;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// Convolution block size; also the extra latency on top of the FIR delay.
pub const BLOCK: usize = 256;
const FFT_LEN: usize = 2 * BLOCK;
const BINS: usize = BLOCK + 1;
const MAX_PARTS: usize = 128;

#[derive(Enum, Clone, Copy, PartialEq, Eq, Debug)]
pub enum LinearLatency {
    #[name = "Low"]
    Low,
    #[name = "Medium"]
    Medium,
    #[name = "High"]
    High,
}

impl LinearLatency {
    /// FIR length in taps; longer means finer low-frequency resolution but more latency.
    pub fn fir_len(self) -> usize {
        match self {
            Self::Low => 2048,
            Self::Medium => 8192,
            Self::High => 32768,
        }
    }
    pub fn latency(self) -> usize {
        self.fir_len() / 2 + BLOCK
    }
    fn parts(self) -> usize {
        self.fir_len() / BLOCK
    }
}

/// Partitioned frequency-domain filter matrix. Index = output * 2 + input (0 = L, 1 = R).
pub struct Kernel {
    parts: usize,
    cross: bool,
    latency: usize,
    filters: [Vec<Complex<f32>>; 4],
}

pub struct LinearTask {
    pub bands: [BandSettings; NUM_BANDS],
    pub quality: LinearLatency,
    pub sr: f32,
}

#[derive(Default)]
pub struct Shared {
    ready: Mutex<Option<Box<Kernel>>>,
    recycle: Mutex<Option<Box<Kernel>>>,
    pub pending: AtomicBool,
}

impl Shared {
    /// Background thread: build and publish a kernel, freeing any returned one.
    pub fn run_task(&self, task: &LinearTask) {
        let kernel = design_kernel(&task.bands, task.quality, task.sr as f64);
        if let Ok(mut r) = self.recycle.lock() {
            r.take();
        }
        if let Ok(mut r) = self.ready.lock() {
            // Any kernel the audio thread never picked up is freed here, off the audio thread.
            *r = Some(kernel);
        }
    }
}

pub fn design_kernel(bands: &[BandSettings; NUM_BANDS], quality: LinearLatency, sr: f64) -> Box<Kernel> {
    let n = quality.fir_len();
    let nb = n / 2 + 1;
    let (mut ll, mut lr, mut rl, mut rr) = (vec![1.0f64; nb], vec![0.0f64; nb], vec![0.0f64; nb], vec![1.0f64; nb]);
    let mut mag = vec![0.0f64; nb];

    for s in bands.iter().filter(|s| s.enabled) {
        eq::band_magnitude_grid(s, sr, &mut mag);
        for k in 0..nb {
            let h = mag[k];
            // Band matrix P acting on (L, R); total M = P * M.
            let (p00, p01, p10, p11) = match s.placement {
                Placement::Stereo => (h, 0.0, 0.0, h),
                Placement::Left => (h, 0.0, 0.0, 1.0),
                Placement::Right => (1.0, 0.0, 0.0, h),
                Placement::Mid => ((h + 1.0) * 0.5, (h - 1.0) * 0.5, (h - 1.0) * 0.5, (h + 1.0) * 0.5),
                Placement::Side => ((1.0 + h) * 0.5, (1.0 - h) * 0.5, (1.0 - h) * 0.5, (1.0 + h) * 0.5),
            };
            let (a, b, c, d) = (ll[k], lr[k], rl[k], rr[k]);
            ll[k] = p00 * a + p01 * c;
            lr[k] = p00 * b + p01 * d;
            rl[k] = p10 * a + p11 * c;
            rr[k] = p10 * b + p11 * d;
        }
    }
    let cross = lr.iter().chain(rl.iter()).any(|v| v.abs() > 1e-7);

    let mut planner = RealFftPlanner::<f32>::new();
    let inv = planner.plan_fft_inverse(n);
    let fwd = planner.plan_fft_forward(FFT_LEN);
    let parts = quality.parts();
    let mut filters: [Vec<Complex<f32>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    let mut spec = inv.make_input_vec();
    let mut ir = inv.make_output_vec();
    let mut scratch_i = inv.make_scratch_vec();
    let mut time = fwd.make_input_vec();
    let mut scratch_f = fwd.make_scratch_vec();

    for (idx, mags) in [&ll, &lr, &rl, &rr].into_iter().enumerate() {
        if !cross && (idx == 1 || idx == 2) {
            continue;
        }
        for (s, &m) in spec.iter_mut().zip(mags.iter()) {
            *s = Complex::new(m as f32, 0.0);
        }
        spec[0].im = 0.0;
        spec[nb - 1].im = 0.0;
        if inv.process_with_scratch(&mut spec, &mut ir, &mut scratch_i).is_err() {
            continue;
        }
        // Zero-phase IR -> centred, Hann-windowed taps.
        let taps: Vec<f32> = (0..n)
            .map(|i| {
                let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos();
                ir[(i + n / 2) % n] / n as f32 * w
            })
            .collect();
        let mut out = vec![Complex::new(0.0, 0.0); parts * BINS];
        for p in 0..parts {
            time.fill(0.0);
            time[..BLOCK].copy_from_slice(&taps[p * BLOCK..(p + 1) * BLOCK]);
            let dst = &mut out[p * BINS..(p + 1) * BINS];
            let _ = fwd.process_with_scratch(&mut time, dst, &mut scratch_f);
        }
        filters[idx] = out;
    }
    Box::new(Kernel { parts, cross, latency: quality.latency(), filters })
}

pub struct Convolver {
    shared: Arc<Shared>,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    active: Option<Box<Kernel>>,
    fading: Option<Box<Kernel>>,
    garbage: Option<Box<Kernel>>,
    fdl: [Vec<Complex<f32>>; 2],
    head: usize,
    win: [Vec<f32>; 2],
    out: [Vec<f32>; 2],
    pos: usize,
    time: Vec<f32>,
    acc: Vec<Complex<f32>>,
    old: Vec<f32>,
    scratch_f: Vec<Complex<f32>>,
    scratch_i: Vec<Complex<f32>>,
}

impl Convolver {
    pub fn new(shared: Arc<Shared>) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(FFT_LEN);
        let inv = planner.plan_fft_inverse(FFT_LEN);
        Self {
            shared,
            scratch_f: fwd.make_scratch_vec(),
            scratch_i: inv.make_scratch_vec(),
            fwd,
            inv,
            active: None,
            fading: None,
            garbage: None,
            fdl: std::array::from_fn(|_| vec![Complex::new(0.0, 0.0); MAX_PARTS * BINS]),
            head: 0,
            win: std::array::from_fn(|_| vec![0.0; FFT_LEN]),
            out: std::array::from_fn(|_| vec![0.0; BLOCK]),
            pos: 0,
            time: vec![0.0; FFT_LEN],
            acc: vec![Complex::new(0.0, 0.0); BINS],
            old: vec![0.0; BLOCK],
        }
    }

    /// Not realtime-safe: drops kernels. Call from `initialize`.
    pub fn clear(&mut self) {
        self.active = None;
        self.fading = None;
        self.garbage = None;
        self.reset();
    }

    pub fn reset(&mut self) {
        for ch in 0..2 {
            self.fdl[ch].fill(Complex::new(0.0, 0.0));
            self.win[ch].fill(0.0);
            self.out[ch].fill(0.0);
        }
        self.pos = 0;
        self.head = 0;
    }

    /// Latency of the kernel currently in use, if any.
    pub fn active_latency(&self) -> Option<usize> {
        self.active.as_ref().map(|k| k.latency)
    }

    #[inline]
    pub fn process_sample(&mut self, l: f32, r: f32) -> (f32, f32) {
        let p = self.pos;
        let out = (self.out[0][p], self.out[1][p]);
        self.win[0][BLOCK + p] = l;
        self.win[1][BLOCK + p] = r;
        self.pos += 1;
        if self.pos == BLOCK {
            self.pos = 0;
            self.hop();
        }
        if self.active.is_some() {
            out
        } else {
            (l, r)
        }
    }

    fn hop(&mut self) {
        // Recycle the previous fading kernel to the background thread; never drop it here.
        if let Some(g) = self.garbage.take() {
            let mut leftover = Some(g);
            if let Ok(mut r) = self.shared.recycle.try_lock() {
                if r.is_none() {
                    *r = leftover.take();
                }
            }
            self.garbage = leftover;
        }
        if self.garbage.is_none() && self.fading.is_none() {
            if let Ok(mut g) = self.shared.ready.try_lock() {
                if let Some(new) = g.take() {
                    self.fading = self.active.replace(new);
                }
            }
        }

        self.head = (self.head + 1) % MAX_PARTS;
        for ch in 0..2 {
            self.time.copy_from_slice(&self.win[ch]);
            let dst = &mut self.fdl[ch][self.head * BINS..(self.head + 1) * BINS];
            let _ = self.fwd.process_with_scratch(&mut self.time, dst, &mut self.scratch_f);
            self.win[ch].copy_within(BLOCK..FFT_LEN, 0);
        }

        if let Some(k) = self.active.as_deref() {
            for o in 0..2 {
                convolve(k, &self.fdl, self.head, o, &mut self.acc, &mut self.time, &*self.inv, &mut self.scratch_i);
                self.out[o].copy_from_slice(&self.time[BLOCK..]);
                if let Some(old) = self.fading.as_deref() {
                    convolve(old, &self.fdl, self.head, o, &mut self.acc, &mut self.time, &*self.inv, &mut self.scratch_i);
                    self.old.copy_from_slice(&self.time[BLOCK..]);
                    for i in 0..BLOCK {
                        let t = (i + 1) as f32 / BLOCK as f32;
                        self.out[o][i] = self.old[i] * (1.0 - t) + self.out[o][i] * t;
                    }
                }
                let norm = 1.0 / FFT_LEN as f32;
                self.out[o].iter_mut().for_each(|v| *v *= norm);
            }
        }
        if let Some(f) = self.fading.take() {
            self.garbage = Some(f);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn convolve(
    k: &Kernel,
    fdl: &[Vec<Complex<f32>>; 2],
    head: usize,
    o: usize,
    acc: &mut [Complex<f32>],
    time: &mut [f32],
    inv: &dyn ComplexToReal<f32>,
    scratch: &mut [Complex<f32>],
) {
    acc.fill(Complex::new(0.0, 0.0));
    let inputs = if k.cross { 0..2 } else { o..o + 1 };
    for input in inputs {
        let h = &k.filters[o * 2 + input];
        for p in 0..k.parts {
            let idx = (head + MAX_PARTS - p) % MAX_PARTS;
            let x = &fdl[input][idx * BINS..(idx + 1) * BINS];
            let hp = &h[p * BINS..(p + 1) * BINS];
            for ((a, x), h) in acc.iter_mut().zip(x).zip(hp) {
                a.re += x.re * h.re - x.im * h.im;
                a.im += x.re * h.im + x.im * h.re;
            }
        }
    }
    acc[0].im = 0.0;
    acc[BINS - 1].im = 0.0;
    let _ = inv.process_with_scratch(acc, time, scratch);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(quality: LinearLatency, bands: [BandSettings; NUM_BANDS], freq: f64) -> f32 {
        let sr = 48000.0;
        let shared = Arc::new(Shared::default());
        shared.run_task(&LinearTask { bands, quality, sr });
        let mut c = Convolver::new(shared);
        let lat = quality.latency();
        let (mut sq, mut cnt) = (0.0f64, 0usize);
        let total = lat * 3 + 4096;
        for n in 0..total {
            let x = (std::f64::consts::TAU * freq * n as f64 / sr as f64).sin() as f32;
            let (y, _) = c.process_sample(x, x);
            if n > lat * 2 + 1024 {
                sq += (y * y) as f64;
                cnt += 1;
            }
        }
        assert_eq!(c.active_latency(), Some(lat));
        ((sq / cnt as f64).sqrt() * std::f64::consts::SQRT_2) as f32
    }

    #[test]
    fn linear_phase_matches_magnitude_and_latency() {
        let mut b = [BandSettings::default(); NUM_BANDS];
        b[0] = BandSettings { enabled: true, freq: 1000.0, gain_db: 12.0, q: 1.0, ..Default::default() };
        let db = |p: f32| 20.0 * p.log10();
        for (q, f) in [(LinearLatency::Low, 1000.0), (LinearLatency::Low, 16000.0), (LinearLatency::Medium, 1000.0)] {
            let want = eq::magnitude_response_db(&b, 48000.0, f, eq::ResponseView::Left) as f32;
            let got = db(run(q, b, f));
            assert!((got - want).abs() < 0.4, "{f} Hz: got {got} want {want}");
        }
    }

    #[test]
    fn linear_phase_is_symmetric_delay() {
        // Impulse through a flat EQ comes out exactly `latency` samples later.
        let quality = LinearLatency::Low;
        let shared = Arc::new(Shared::default());
        shared.run_task(&LinearTask { bands: [BandSettings::default(); NUM_BANDS], quality, sr: 48000.0 });
        let mut c = Convolver::new(shared);
        let lat = quality.latency();
        let mut best = (0usize, 0.0f32);
        for n in 0..lat * 2 + 600 {
            let x = if n == 700 { 1.0 } else { 0.0 };
            let (y, _) = c.process_sample(x, 0.0);
            if y.abs() > best.1 {
                best = (n, y.abs());
            }
        }
        // Skip the dry passthrough before the first kernel is swapped in.
        assert_eq!(best.0, 700 + lat);
    }

    #[test]
    fn mid_band_leaves_side_untouched() {
        let mut b = [BandSettings::default(); NUM_BANDS];
        b[0] = BandSettings {
            enabled: true,
            freq: 1000.0,
            gain_db: -12.0,
            q: 1.0,
            placement: Placement::Mid,
            ..Default::default()
        };
        let k = design_kernel(&b, LinearLatency::Low, 48000.0);
        assert!(k.cross);
    }
}
