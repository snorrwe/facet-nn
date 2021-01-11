use criterion::{criterion_group, Criterion};
use criterion::{criterion_main, BenchmarkId};

use facet_core::gpu::matmul::matmul_f64_impl;
use facet_core::ndarray::matrix::matmul_impl;
use facet_core::ndarray::NdArray;
use rand::Rng;

fn mat_mul(c: &mut Criterion) {
    let mut g = c.benchmark_group("matrix multiplication");
    for size in (256..=542).step_by(27) {
        g.bench_with_input(
            BenchmarkId::new("cpu", size),
            &size,
            move |bencher, &size| {
                let mut rng = rand::thread_rng();
                let mut a = NdArray::new([size, size]);
                let mut b = NdArray::new([size, size]);
                let mut c = NdArray::new([size, size]);
                for i in 0..size as usize * size as usize {
                    a.as_mut_slice()[i] = rng.gen_range(-1.2f64, 1.2);
                    b.as_mut_slice()[i] = rng.gen_range(-1.2f64, 1.2);
                }

                bencher.iter(move || {
                    matmul_impl([size; 3], a.as_slice(), b.as_slice(), c.as_mut_slice())
                })
            },
        );
        g.bench_with_input(
            BenchmarkId::new("gpu", size),
            &size,
            move |bencher, &size| {
                let mut rng = rand::thread_rng();
                let mut a = NdArray::new([size, size]);
                let mut b = NdArray::new([size, size]);
                let mut c = NdArray::new([size, size]);
                for i in 0..size as usize * size as usize {
                    a.as_mut_slice()[i] = rng.gen_range(-1.2f64, 1.2);
                    b.as_mut_slice()[i] = rng.gen_range(-1.2f64, 1.2);
                }

                bencher.iter(move || {
                    matmul_f64_impl([size; 3], a.as_slice(), b.as_slice(), c.as_mut_slice())
                })
            },
        );
    }
    g.finish();
}

criterion_group!(matrices, mat_mul,);

criterion_main!(matrices);
