use criterion::{criterion_group, Criterion};
use criterion::{criterion_main, BenchmarkId};

use facet_core::gpu::matmul::matmul_f32_impl;
use facet_core::ndarray::matrix::matmul_impl;
use facet_core::ndarray::NdArray;
use rand::Rng;

fn _abc(size: u32) -> [NdArray<f32>; 3] {
    let mut rng = rand::thread_rng();
    let mut a = NdArray::new([size, size]);
    let mut b = NdArray::new([size, size]);
    let c = NdArray::new([size, size]);
    for i in 0..size as usize * size as usize {
        a.as_mut_slice()[i] = rng.gen_range(-1.2f32, 1.2);
        b.as_mut_slice()[i] = rng.gen_range(-1.2f32, 1.2);
    }

    [a, b, c]
}

fn mat_mul(c: &mut Criterion) {
    let mut g = c.benchmark_group("matrix multiplication");
    for size in (16..=550).step_by(128) {
        g.bench_with_input(
            BenchmarkId::new("cpu", size),
            &size,
            move |bencher, &size| {
                let [a, b, mut c] = _abc(size);
                bencher.iter(move || {
                    matmul_impl([size; 3], a.as_slice(), b.as_slice(), c.as_mut_slice()).unwrap()
                })
            },
        );
        g.bench_with_input(
            BenchmarkId::new("gpu", size),
            &size,
            move |bencher, &size| {
                let [a, b, mut c] = _abc(size);
                bencher.iter(move || {
                    matmul_f32_impl([size; 3], a.as_slice(), b.as_slice(), c.as_mut_slice())
                        .unwrap()
                })
            },
        );
    }
    g.finish();
}

criterion_group!(matrices, mat_mul,);

criterion_main!(matrices);
