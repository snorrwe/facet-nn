use criterion::criterion_main;
use facet_core::ndarray::NdArray;
use rand::Rng;

fn _random_mat(cols: u32, rows: u32) -> NdArray<f32> {
    let mut rng = rand::thread_rng();
    NdArray::new_with_values(
        [cols, rows],
        (0..cols as usize * rows as usize)
            .map(|_| rng.gen_range(-1.0f32, 1.0))
            .collect(),
    )
    .unwrap()
}

/// return 3 square matrices of size `size`
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

mod mat_mul {
    use super::_abc;
    use criterion::{criterion_group, BenchmarkId, Criterion};
    use facet_core::gpu::matmul::matmul_a_mul_b::matmul_f32_impl;
    use facet_core::ndarray::matrix::matmul_impl;

    fn mat_mul(c: &mut Criterion) {
        let mut g = c.benchmark_group("matrix multiplication");
        for size in (16..=550).step_by(128) {
            g.bench_with_input(
                BenchmarkId::new("cpu", size),
                &size,
                move |bencher, &size| {
                    let [a, b, mut c] = _abc(size);
                    bencher.iter(move || {
                        matmul_impl([size; 3], a.as_slice(), b.as_slice(), c.as_mut_slice())
                            .unwrap()
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

    criterion_group!(matrices, mat_mul);
}

mod dense_layer {
    use super::_random_mat;
    use criterion::{criterion_group, BenchmarkId, Criterion};
    use facet_core::layer::dense_layer::DenseLayer;

    fn backward(c: &mut Criterion) {
        let mut g = c.benchmark_group("dense layer backward pass");

        for inp_size in (16..80).step_by(17) {
            g.bench_with_input(
                BenchmarkId::new("backward", inp_size),
                &inp_size,
                move |bencher, &size| {
                    let mut layer =
                        DenseLayer::new(size, size).with_training(None, None, None, None);

                    let x = _random_mat(size * 500, size);

                    bencher.iter(move || {
                        layer.forward(x.clone()).unwrap();
                        layer.backward(layer.output.clone()).unwrap()
                    })
                },
            );
        }
    }
    criterion_group!(dense_layer, backward);
}

criterion_main!(mat_mul::matrices, dense_layer::dense_layer);
