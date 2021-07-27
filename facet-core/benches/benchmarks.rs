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

criterion_main!(dense_layer::dense_layer);
