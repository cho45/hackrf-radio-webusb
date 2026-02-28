use std::hint::black_box;
use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use num_complex::Complex;

use hackrf_dsp::bench_filter::{design_lowpass_hamming_coeffs, ComplexFirFilter};

const BLOCK_LEN: usize = 200_000;

fn build_input(seed: u32, len: usize) -> Vec<Complex<f32>> {
    let mut state = seed;
    let mut input = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u0 = ((state >> 8) as f32) * (1.0 / 16_777_216.0);
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u1 = ((state >> 8) as f32) * (1.0 / 16_777_216.0);
        input.push(Complex::new(u0 * 2.0 - 1.0, u1 * 2.0 - 1.0));
    }
    input
}

struct RealCoeffFixture {
    filter: ComplexFirFilter,
    input: Vec<Complex<f32>>,
}

impl RealCoeffFixture {
    fn new(taps: usize, cutoff: f32, seed: u32) -> Self {
        Self {
            filter: ComplexFirFilter::new_lowpass_hamming(taps, cutoff),
            input: build_input(seed, BLOCK_LEN),
        }
    }

    fn run_once(&mut self) -> f32 {
        self.filter.reset();
        let mut sink = 0.0f32;
        for &x in &self.input {
            let y = self.filter.process_sample(x);
            sink += y.re * 1e-12 + y.im * 1e-12;
        }
        sink
    }
}

struct ComplexCoeffFixture {
    filter: ComplexFirFilter,
    input: Vec<Complex<f32>>,
}

impl ComplexCoeffFixture {
    fn new(taps: usize, cutoff: f32, seed: u32) -> Self {
        let coeffs = design_lowpass_hamming_coeffs(taps, cutoff)
            .into_iter()
            .map(|h| Complex::new(h, 0.0))
            .collect::<Vec<_>>();
        Self {
            filter: ComplexFirFilter::new_complex_coeffs(coeffs),
            input: build_input(seed, BLOCK_LEN),
        }
    }

    fn run_once(&mut self) -> f32 {
        self.filter.reset();
        let mut sink = 0.0f32;
        for &x in &self.input {
            let y = self.filter.process_sample(x);
            sink += y.re * 1e-12 + y.im * 1e-12;
        }
        sink
    }
}

fn bench_complex_fir_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_fir_variants");
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    group.noise_threshold(0.03);
    group.significance_level(0.01);
    group.confidence_level(0.99);
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(BLOCK_LEN as u64));

    let cutoff = 15_000.0f32 / 200_000.0f32;
    for taps in [63usize, 127, 255] {
        let mut fixtures_real: Vec<RealCoeffFixture> = (0..8)
            .map(|i| RealCoeffFixture::new(taps, cutoff, 0x1000_0000u32 + i))
            .collect();
        let mut cursor_real = 0usize;
        group.bench_with_input(BenchmarkId::new("real_coeff", taps), &taps, |b, _| {
            b.iter(|| {
                let fixture = &mut fixtures_real[cursor_real];
                cursor_real = (cursor_real + 1) & 7;
                black_box(fixture.run_once());
            });
        });

        let mut fixtures_cplx: Vec<ComplexCoeffFixture> = (0..8)
            .map(|i| ComplexCoeffFixture::new(taps, cutoff, 0x2000_0000u32 + i))
            .collect();
        let mut cursor_cplx = 0usize;
        group.bench_with_input(
            BenchmarkId::new("complex_coeff(real_h)", taps),
            &taps,
            |b, _| {
                b.iter(|| {
                    let fixture = &mut fixtures_cplx[cursor_cplx];
                    cursor_cplx = (cursor_cplx + 1) & 7;
                    black_box(fixture.run_once());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_complex_fir_variants);
criterion_main!(benches);
