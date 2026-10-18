//! Spectrum analyzer: a lock-free tap on the audio thread plus the GUI-side FFT.

use crate::eq::{MAX_FREQ, MIN_FREQ};
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

const RING_LEN: usize = 1 << 16;
pub const COLUMNS: usize = 640;
pub const FLOOR_DB: f32 = -120.0;
/// Spectrum tilt in dB per octave, relative to 1 kHz.
pub const TILT_DB_PER_OCT: f32 = 4.5;

/// Single-producer single-consumer ring of f32 samples. The producer (audio
/// thread) never blocks or allocates; the consumer detects overruns by position.
pub struct Ring {
    buf: Box<[AtomicU32]>,
    head: AtomicUsize,
}

impl Ring {
    fn new() -> Self {
        Self {
            buf: (0..RING_LEN).map(|_| AtomicU32::new(0)).collect(),
            head: AtomicUsize::new(0),
        }
    }

    /// Audio thread only (single producer).
    #[inline]
    pub fn push(&self, x: f32) {
        let h = self.head.load(Ordering::Relaxed);
        self.buf[h & (RING_LEN - 1)].store(x.to_bits(), Ordering::Relaxed);
        self.head.store(h.wrapping_add(1), Ordering::Release);
    }

    fn head(&self) -> usize {
        self.head.load(Ordering::Acquire)
    }
}

/// Shared between the audio thread (writer) and the GUI (reader).
pub struct SpectrumTap {
    pub pre: Ring,
    pub post: Ring,
}

impl SpectrumTap {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { pre: Ring::new(), post: Ring::new() })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpectrumMode {
    Off,
    Pre,
    Post,
    Both,
}

impl SpectrumMode {
    pub fn pre(self) -> bool {
        matches!(self, Self::Pre | Self::Both)
    }
    pub fn post(self) -> bool {
        matches!(self, Self::Post | Self::Both)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resolution {
    Low,
    Medium,
    High,
}

impl Resolution {
    pub fn fft_size(self) -> usize {
        match self {
            Self::Low => 2048,
            Self::Medium => 4096,
            Self::High => 8192,
        }
    }
}

struct Reader {
    pos: usize,
    hist: Vec<f32>,
}

impl Reader {
    fn new(n: usize, ring: &Ring) -> Self {
        Self { pos: ring.head(), hist: vec![0.0; n] }
    }

    fn skip_to_head(&mut self, ring: &Ring) {
        self.pos = ring.head();
    }

    /// Advance by one hop if enough new samples exist. Resyncs after overruns.
    fn advance(&mut self, ring: &Ring, hop: usize) -> bool {
        let head = ring.head();
        let avail = head.wrapping_sub(self.pos);
        if avail > RING_LEN / 2 {
            self.pos = head.wrapping_sub(hop);
        } else if avail < hop {
            return false;
        }
        let n = self.hist.len();
        self.hist.copy_within(hop.., 0);
        for i in 0..hop {
            let v = f32::from_bits(ring.buf[self.pos.wrapping_add(i) & (RING_LEN - 1)].load(Ordering::Relaxed));
            self.hist[n - hop + i] = v;
        }
        self.pos = self.pos.wrapping_add(hop);
        true
    }
}

/// One analysed channel (pre or post) on the display columns.
pub struct Trace {
    pub smooth: Vec<f32>,
    pub peak: Vec<f32>,
    reader: Reader,
}

pub struct Analyzer {
    tap: Arc<SpectrumTap>,
    size: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    window_sum: f32,
    input: Vec<f32>,
    output: Vec<realfft::num_complex::Complex<f32>>,
    scratch: Vec<realfft::num_complex::Complex<f32>>,
    power: Vec<f32>,
    pub pre: Trace,
    pub post: Trace,
    last_sr: f64,
}

impl Analyzer {
    pub fn new(tap: Arc<SpectrumTap>) -> Self {
        let mut a = Self::build(tap, Resolution::Medium.fft_size());
        a.last_sr = 44100.0;
        a
    }

    fn build(tap: Arc<SpectrumTap>, size: usize) -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(size);
        let window: Vec<f32> = (0..size)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / size as f32).cos())
            .collect();
        let trace = |ring: &Ring| Trace {
            smooth: vec![FLOOR_DB; COLUMNS],
            peak: vec![FLOOR_DB; COLUMNS],
            reader: Reader::new(size, ring),
        };
        Self {
            input: fft.make_input_vec(),
            output: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            power: vec![0.0; size / 2 + 1],
            window_sum: window.iter().sum(),
            window,
            fft,
            pre: trace(&tap.pre),
            post: trace(&tap.post),
            tap,
            size,
            last_sr: 44100.0,
        }
    }

    pub fn set_resolution(&mut self, res: Resolution) {
        if res.fft_size() != self.size {
            *self = Self::build(self.tap.clone(), res.fft_size());
        }
    }

    /// Consume pending audio and update traces. Call once per GUI frame.
    pub fn update(&mut self, mode: SpectrumMode, sr: f64) {
        self.last_sr = sr;
        let hop = self.size / 2;
        let dt = hop as f32 / sr as f32;
        let attack = 1.0 - (-dt / 0.012).exp();
        let release = 1.0 - (-dt / 0.350).exp();
        let peak_fall = 6.0 * dt; // dB per hop; ~6 dB/s
        let size = self.size;

        for which in 0..2 {
            let enabled = if which == 0 { mode.pre() } else { mode.post() };
            let ring = if which == 0 { &self.tap.pre } else { &self.tap.post };
            let trace = if which == 0 { &mut self.pre } else { &mut self.post };
            if !enabled {
                trace.reader.skip_to_head(ring);
                continue;
            }
            // Bound the work per frame; skip ahead if the GUI fell behind.
            for _ in 0..8 {
                if !trace.reader.advance(ring, hop) {
                    break;
                }
                for i in 0..size {
                    self.input[i] = trace.reader.hist[i] * self.window[i];
                }
                if self
                    .fft
                    .process_with_scratch(&mut self.input, &mut self.output, &mut self.scratch)
                    .is_err()
                {
                    break;
                }
                let norm = 2.0 / self.window_sum;
                for (p, c) in self.power.iter_mut().zip(self.output.iter()) {
                    let m = c.norm() * norm;
                    *p = m * m;
                }
                map_to_columns(&self.power, size, sr, trace, attack, release, peak_fall);
            }
        }
    }

    pub fn sample_rate(&self) -> f64 {
        self.last_sr
    }
}

pub fn column_freq(edge: f32) -> f64 {
    // edge in 0..=COLUMNS
    MIN_FREQ * (MAX_FREQ / MIN_FREQ).powf(edge as f64 / COLUMNS as f64)
}

fn map_to_columns(
    power: &[f32],
    size: usize,
    sr: f64,
    trace: &mut Trace,
    attack: f32,
    release: f32,
    peak_fall: f32,
) {
    let bin_hz = sr / size as f64;
    let nyq_bin = power.len() - 1;
    let db_of = |p: f32| 10.0 * (p + 1e-12).log10();
    for c in 0..COLUMNS {
        let (f0, f1) = (column_freq(c as f32), column_freq(c as f32 + 1.0));
        let (b0, b1) = (f0 / bin_hz, f1 / bin_hz);
        let centre = (f0 * f1).sqrt();
        let mut db = if b0 >= nyq_bin as f64 {
            FLOOR_DB
        } else if b1 - b0 < 1.0 {
            // Narrow column: interpolate between neighbouring bins in dB.
            let p = ((b0 + b1) * 0.5).min(nyq_bin as f64);
            let lo = p.floor() as usize;
            let hi = (lo + 1).min(nyq_bin);
            let t = (p - lo as f64) as f32;
            db_of(power[lo]) * (1.0 - t) + db_of(power[hi]) * t
        } else {
            let lo = b0.ceil() as usize;
            let hi = (b1.floor() as usize).min(nyq_bin).max(lo);
            let sum: f32 = power[lo..=hi].iter().sum();
            db_of(sum / (hi - lo + 1) as f32)
        };
        db += TILT_DB_PER_OCT * (centre / 1000.0).log2() as f32;
        let db = db.max(FLOOR_DB);
        let s = &mut trace.smooth[c];
        *s += (db - *s) * if db > *s { attack } else { release };
        let p = &mut trace.peak[c];
        *p = if *s > *p { *s } else { (*p - peak_fall).max(*s) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_peaks_at_its_frequency() {
        let tap = SpectrumTap::new();
        let mut a = Analyzer::new(tap.clone());
        let sr = 48000.0f64;
        for round in 0..20 {
            for n in 0..4096 {
                let t = (round * 4096 + n) as f64 / sr;
                let x = (std::f64::consts::TAU * 1000.0 * t).sin() as f32;
                tap.post.push(x);
                tap.pre.push(x);
            }
            a.update(SpectrumMode::Both, sr);
        }
        let (best, _) = a
            .post
            .smooth
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap();
        let f = column_freq(best as f32 + 0.5);
        assert!((f - 1000.0).abs() < 100.0, "peak at {f}");
    }
}
