//! Output metering and auto gain (loudness matching).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Written by the audio thread, read by the GUI. Lock-free.
pub struct Levels {
    /// Per-channel peak since the GUI last took it (linear, non-negative f32 bits).
    peak: [AtomicU32; 2],
    auto_gain_db: AtomicU32,
}

impl Levels {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            peak: [AtomicU32::new(0), AtomicU32::new(0)],
            auto_gain_db: AtomicU32::new(0f32.to_bits()),
        })
    }

    /// Audio thread. Non-negative floats order the same as their bit patterns.
    pub fn report(&self, peak_l: f32, peak_r: f32, auto_gain_db: f32) {
        self.peak[0].fetch_max(peak_l.abs().to_bits(), Ordering::Relaxed);
        self.peak[1].fetch_max(peak_r.abs().to_bits(), Ordering::Relaxed);
        self.auto_gain_db.store(auto_gain_db.to_bits(), Ordering::Relaxed);
    }

    /// GUI thread: peaks since the previous call.
    pub fn take_peaks(&self) -> [f32; 2] {
        [
            f32::from_bits(self.peak[0].swap(0, Ordering::Relaxed)),
            f32::from_bits(self.peak[1].swap(0, Ordering::Relaxed)),
        ]
    }

    pub fn auto_gain_db(&self) -> f32 {
        f32::from_bits(self.auto_gain_db.load(Ordering::Relaxed))
    }
}

const MAX_AUTO_GAIN_DB: f64 = 24.0;
const POWER_WINDOW_S: f64 = 0.6;
const GAIN_SMOOTH_S: f64 = 0.25;

/// Matches the loudness (slow RMS) of the EQ output to its input.
pub struct AutoGain {
    p_in: f64,
    p_out: f64,
    gain: f64,
    target: f64,
    c_pow: f64,
    c_gain: f64,
}

impl AutoGain {
    pub fn new() -> Self {
        let mut a = Self { p_in: 0.0, p_out: 0.0, gain: 1.0, target: 1.0, c_pow: 0.0, c_gain: 0.0 };
        a.set_sample_rate(44100.0);
        a
    }

    pub fn set_sample_rate(&mut self, sr: f64) {
        self.c_pow = 1.0 - (-1.0 / (POWER_WINDOW_S * sr)).exp();
        self.c_gain = 1.0 - (-1.0 / (GAIN_SMOOTH_S * sr)).exp();
    }

    pub fn reset(&mut self) {
        self.p_in = 0.0;
        self.p_out = 0.0;
        self.gain = 1.0;
        self.target = 1.0;
    }

    /// Returns the linear gain to apply to the EQ output.
    #[inline]
    pub fn process(&mut self, input: (f32, f32), output: (f32, f32), enabled: bool) -> f32 {
        let pi = 0.5 * (input.0 as f64).powi(2) + 0.5 * (input.1 as f64).powi(2);
        let po = 0.5 * (output.0 as f64).powi(2) + 0.5 * (output.1 as f64).powi(2);
        self.p_in += (pi - self.p_in) * self.c_pow;
        self.p_out += (po - self.p_out) * self.c_pow;
        if enabled {
            // Hold the last target during silence to avoid pumping the noise floor.
            if self.p_in > 1e-10 && self.p_out > 1e-10 {
                let lim = 10f64.powf(MAX_AUTO_GAIN_DB / 20.0);
                self.target = (self.p_in / self.p_out).sqrt().clamp(1.0 / lim, lim);
            }
        } else {
            self.target = 1.0;
        }
        self.gain += (self.target - self.gain) * self.c_gain;
        self.gain as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_gain_matches_loudness() {
        let mut ag = AutoGain::new();
        let mut g = 1.0;
        for n in 0..96000 {
            let x = (n as f32 * 0.1).sin() * 0.25;
            g = ag.process((x, x), (x * 4.0, x * 4.0), true); // EQ boosted +12 dB
        }
        assert!((g - 0.25).abs() < 0.01, "gain {g}");
        for _ in 0..96000 {
            g = ag.process((0.1, 0.1), (0.4, 0.4), false);
        }
        assert!((g - 1.0).abs() < 0.01);
    }
}
