use num_complex::Complex;
use crate::filter::{ComplexCoeffFirFilter, design_lowpass_hamming_coeffs};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SSBMode {
    Usb,
    Lsb,
}

fn build_complex_bandpass_hamming_coeffs(
    num_taps: usize,
    sample_rate_hz: f32,
    center_hz: f32,
    bandwidth_hz: f32,
) -> Vec<Complex<f32>> {
    let cutoff_norm = (bandwidth_hz * 0.5 / sample_rate_hz).clamp(1e-6, 0.49);
    let lp = design_lowpass_hamming_coeffs(num_taps, cutoff_norm);
    let mid = (num_taps - 1) as f32 * 0.5;
    let mut coeffs = Vec::with_capacity(num_taps);
    for (i, &h) in lp.iter().enumerate() {
        let n = i as f32 - mid;
        let phi = 2.0 * std::f32::consts::PI * center_hz * n / sample_rate_hz;
        let (s, c) = phi.sin_cos();
        coeffs.push(Complex::new(h * c, h * s));
    }
    coeffs
}

/// SSB復調器（USB/LSB）
///
/// - 複素I/Qから目的サイドバンドのみを複素帯域通過で抽出
/// - 実部を音声として取り出す
/// - DCブロック + 軽量AGC を適用
pub struct SSBDemodulator {
    sample_rate_hz: f32,
    mode: SSBMode,
    if_min_hz: f32,
    if_max_hz: f32,
    fir_taps: usize,
    sideband_filter: ComplexCoeffFirFilter,
    hp_prev_x: f32,
    hp_prev_y: f32,
    hp_a: f32,
    env: f32,
    env_alpha: f32,
    agc_target: f32,
    agc_max_gain: f32,
}

impl SSBDemodulator {
    pub fn new(sample_rate_hz: f32, mode: SSBMode) -> Self {
        assert!(sample_rate_hz > 0.0, "sample_rate_hz must be > 0");
        let if_min_hz = 300.0f32;
        let if_max_hz = 3_000.0f32;
        let fir_taps = 129usize;
        let sideband_filter = Self::build_sideband_filter(
            sample_rate_hz,
            mode,
            if_min_hz,
            if_max_hz,
            fir_taps,
        );
        Self {
            sample_rate_hz,
            mode,
            if_min_hz,
            if_max_hz,
            fir_taps,
            sideband_filter,
            hp_prev_x: 0.0,
            hp_prev_y: 0.0,
            hp_a: (-2.0 * std::f32::consts::PI * 80.0 / sample_rate_hz).exp(),
            env: 0.0,
            env_alpha: 0.002,
            agc_target: 0.30,
            agc_max_gain: 25.0,
        }
    }

    fn build_sideband_filter(
        sample_rate_hz: f32,
        mode: SSBMode,
        if_min_hz: f32,
        if_max_hz: f32,
        fir_taps: usize,
    ) -> ComplexCoeffFirFilter {
        let min_hz = if_min_hz.max(20.0);
        let max_hz = if_max_hz.max(min_hz + 50.0);
        let center_hz = 0.5 * (min_hz + max_hz);
        let bandwidth_hz = (max_hz - min_hz).max(100.0);
        let signed_center_hz = match mode {
            SSBMode::Usb => center_hz,
            SSBMode::Lsb => -center_hz,
        };
        let coeffs = build_complex_bandpass_hamming_coeffs(
            fir_taps,
            sample_rate_hz,
            signed_center_hz,
            bandwidth_hz,
        );
        ComplexCoeffFirFilter::new(coeffs)
    }

    pub fn set_mode(&mut self, mode: SSBMode) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        self.sideband_filter = Self::build_sideband_filter(
            self.sample_rate_hz,
            self.mode,
            self.if_min_hz,
            self.if_max_hz,
            self.fir_taps,
        );
        self.sideband_filter.reset();
    }

    pub fn set_if_band(&mut self, min_hz: f32, max_hz: f32) {
        self.if_min_hz = min_hz.max(0.0);
        self.if_max_hz = max_hz.max(self.if_min_hz + 50.0);
        self.sideband_filter = Self::build_sideband_filter(
            self.sample_rate_hz,
            self.mode,
            self.if_min_hz,
            self.if_max_hz,
            self.fir_taps,
        );
        self.sideband_filter.reset();
    }

    pub fn reset(&mut self) {
        self.sideband_filter.reset();
        self.hp_prev_x = 0.0;
        self.hp_prev_y = 0.0;
        self.env = 0.0;
    }

    pub fn demodulate(&mut self, input: &[Complex<f32>], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (i, &x) in input.iter().enumerate() {
            let filtered = self.sideband_filter.process_sample(x);
            let audio_raw = filtered.re;

            let hp = audio_raw - self.hp_prev_x + self.hp_a * self.hp_prev_y;
            self.hp_prev_x = audio_raw;
            self.hp_prev_y = hp;

            self.env += self.env_alpha * (hp.abs() - self.env);
            let gain = (self.agc_target / (self.env + 1e-4)).clamp(0.1, self.agc_max_gain);
            output[i] = (hp * gain).clamp(-0.99, 0.99);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone_amp(signal: &[f32], fs_hz: f32, freq_hz: f32) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }
        let w = 2.0 * std::f32::consts::PI * freq_hz / fs_hz;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (n, &x) in signal.iter().enumerate() {
            let phi = w * n as f32;
            re += x * phi.cos();
            im -= x * phi.sin();
        }
        2.0 * (re.hypot(im)) / signal.len() as f32
    }

    fn make_usb_tone(fs_hz: f32, len: usize, hz: f32) -> Vec<Complex<f32>> {
        (0..len)
            .map(|i| {
                let t = i as f32 / fs_hz;
                let phi = 2.0 * std::f32::consts::PI * hz * t;
                Complex::new(phi.cos(), phi.sin())
            })
            .collect()
    }

    fn make_lsb_tone(fs_hz: f32, len: usize, hz: f32) -> Vec<Complex<f32>> {
        (0..len)
            .map(|i| {
                let t = i as f32 / fs_hz;
                let phi = 2.0 * std::f32::consts::PI * hz * t;
                Complex::new(phi.cos(), -phi.sin())
            })
            .collect()
    }

    #[test]
    fn usb_tone_is_recovered() {
        let fs = 50_000.0f32;
        let n = 120_000usize;
        let input = make_usb_tone(fs, n, 1_200.0);
        let mut out = vec![0.0f32; n];
        let mut demod = SSBDemodulator::new(fs, SSBMode::Usb);
        demod.demodulate(&input, &mut out);

        let tail = &out[n / 3..];
        let a1200 = tone_amp(tail, fs, 1_200.0);
        assert!(a1200 > 0.05, "USB tone too small: {}", a1200);
    }

    #[test]
    fn lsb_tone_is_recovered() {
        let fs = 50_000.0f32;
        let n = 120_000usize;
        let input = make_lsb_tone(fs, n, 1_500.0);
        let mut out = vec![0.0f32; n];
        let mut demod = SSBDemodulator::new(fs, SSBMode::Lsb);
        demod.demodulate(&input, &mut out);

        let tail = &out[n / 3..];
        let a1500 = tone_amp(tail, fs, 1_500.0);
        assert!(a1500 > 0.05, "LSB tone too small: {}", a1500);
    }

    #[test]
    fn sideband_rejection_is_reasonable() {
        let fs = 50_000.0f32;
        let n = 160_000usize;
        let usb = make_usb_tone(fs, n, 1_100.0);
        let lsb = make_lsb_tone(fs, n, 1_900.0);
        let mut input = vec![Complex::new(0.0f32, 0.0f32); n];
        for i in 0..n {
            input[i] = usb[i] + lsb[i];
        }

        let mut out = vec![0.0f32; n];
        let mut demod = SSBDemodulator::new(fs, SSBMode::Usb);
        demod.demodulate(&input, &mut out);
        let tail = &out[n / 2..];
        let main = tone_amp(tail, fs, 1_100.0).max(1e-8);
        let leak = tone_amp(tail, fs, 1_900.0).max(1e-8);
        let rej_db = 20.0 * (main / leak).log10();
        assert!(
            rej_db > 20.0,
            "opposite sideband rejection too low: {} dB",
            rej_db
        );
    }
}
