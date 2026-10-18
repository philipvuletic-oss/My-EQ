//! 2x oversampling with a linear-phase Kaiser-windowed sinc FIR, used by Natural Phase mode.
//! Everything is preallocated; processing never allocates.

use std::f64::consts::PI;
use std::sync::Arc;

pub const TAPS: usize = 95;
const HALF: usize = TAPS / 2 + 1;
/// Round-trip latency in samples at the base sample rate (up + down).
pub const LATENCY: usize = (TAPS - 1) / 2;

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..40 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
    }
    sum
}

/// Lowpass with cutoff at the base-rate Nyquist (a quarter of the 2x rate), unity DC gain.
fn design_taps() -> Arc<[f32]> {
    let beta = 8.0;
    let m = (TAPS - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..TAPS)
        .map(|n| {
            let x = n as f64 - m;
            let sinc = if x == 0.0 { 0.5 } else { (0.5 * PI * x).sin() / (PI * x) };
            let r = x / m;
            sinc * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / bessel_i0(beta)
        })
        .collect();
    let sum: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= sum);
    h.iter().map(|&v| v as f32).collect::<Vec<_>>().into()
}

struct Up {
    h: Arc<[f32]>,
    hist: [f32; HALF],
}

impl Up {
    #[inline]
    fn process(&mut self, x: f32) -> (f32, f32) {
        self.hist.copy_within(0..HALF - 1, 1);
        self.hist[0] = x;
        let (mut even, mut odd) = (0.0f32, 0.0f32);
        for k in 0..HALF {
            even += self.h[2 * k] * self.hist[k];
            if 2 * k + 1 < TAPS {
                odd += self.h[2 * k + 1] * self.hist[k];
            }
        }
        (even * 2.0, odd * 2.0)
    }
}

struct Down {
    h: Arc<[f32]>,
    hist: [f32; TAPS],
}

impl Down {
    /// `a` then `b` are two consecutive samples at the 2x rate.
    #[inline]
    fn process(&mut self, a: f32, b: f32) -> f32 {
        self.hist.copy_within(0..TAPS - 1, 1);
        self.hist[0] = a;
        let y: f32 = self.h.iter().zip(self.hist.iter()).map(|(h, x)| h * x).sum();
        self.hist.copy_within(0..TAPS - 1, 1);
        self.hist[0] = b;
        y
    }
}

pub struct StereoUp([Up; 2]);
pub struct StereoDown([Down; 2]);

impl StereoUp {
    pub fn new() -> Self {
        let h = design_taps();
        Self(std::array::from_fn(|_| Up { h: h.clone(), hist: [0.0; HALF] }))
    }
    pub fn reset(&mut self) {
        self.0.iter_mut().for_each(|u| u.hist = [0.0; HALF]);
    }
    /// Returns two 2x-rate stereo samples.
    #[inline]
    pub fn process(&mut self, l: f32, r: f32) -> [(f32, f32); 2] {
        let (l0, l1) = self.0[0].process(l);
        let (r0, r1) = self.0[1].process(r);
        [(l0, r0), (l1, r1)]
    }
}

impl StereoDown {
    pub fn new() -> Self {
        let h = design_taps();
        Self(std::array::from_fn(|_| Down { h: h.clone(), hist: [0.0; TAPS] }))
    }
    pub fn reset(&mut self) {
        self.0.iter_mut().for_each(|d| d.hist = [0.0; TAPS]);
    }
    #[inline]
    pub fn process(&mut self, a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
        (self.0[0].process(a.0, b.0), self.0[1].process(a.1, b.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_flat_and_delayed() {
        let sr = 48000.0f64;
        for f in [100.0, 5000.0, 15000.0, 19000.0] {
            let (mut up, mut down) = (StereoUp::new(), StereoDown::new());
            let (mut peak, mut best_err) = (0.0f32, f32::MAX);
            let n_total = 8000;
            let sig = |n: usize| (2.0 * PI * f * n as f64 / sr).sin() as f32;
            for n in 0..n_total {
                let [a, b] = up.process(sig(n), sig(n));
                let (y, _) = down.process(a, b);
                if n > 2000 {
                    peak = peak.max(y.abs());
                    best_err = best_err.min((y - sig(n - LATENCY)).abs().min(1.0));
                }
            }
            assert!((peak - 1.0).abs() < 0.06, "{f} Hz peak {peak}");
            let _ = best_err;
            // Check the delay exactly on a final stretch.
            let (mut up, mut down) = (StereoUp::new(), StereoDown::new());
            let mut max_err = 0.0f32;
            for n in 0..n_total {
                let [a, b] = up.process(sig(n), sig(n));
                let (y, _) = down.process(a, b);
                if n > 3000 {
                    max_err = max_err.max((y - sig(n - LATENCY)).abs());
                }
            }
            assert!(max_err < 0.07, "{f} Hz delay/gain error {max_err}");
        }
    }
}
