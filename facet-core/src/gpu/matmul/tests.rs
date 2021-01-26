use super::matmul_a_mul_b::*;
use crate::prelude::*;
use rand::Rng;

#[test]
fn test_identity_returns_the_og_matrix() {
    const N: u32 = LOCAL_SIZE_X + 11; // force combined GPU+CPU computation

    let mut rng = rand::thread_rng();
    let a = NdArray::new_with_values([N, N], (0..N * N).map(|_| rng.gen_range(0., 10.)).collect())
        .unwrap();

    let mut b = NdArray::new_with_values(&[N, N][..], (0..N * N).map(|_| 0.0).collect()).unwrap();
    for i in 0..N {
        for j in 0..N {
            if i == j {
                *b.get_mut(&[i, j]).unwrap() = 1.0
            }
        }
    }

    let mut c_gpu = NdArray::new([N, N]);
    matmul_f32_impl([N, N, N], a.as_slice(), b.as_slice(), c_gpu.as_mut_slice()).unwrap();

    println!("{}", c_gpu);

    for (c, a) in c_gpu
        .as_slice()
        .iter()
        .zip(a.as_slice().iter())
        .map(|(c, a)| (*c, *a))
    {
        assert!((a - c).abs() < 0.001)
    }
}

#[test]
fn test_correctness() {
    const N: u32 = LOCAL_SIZE_X + 11; // force combined GPU+CPU computation
    const M: u32 = 70;
    const P: u32 = LOCAL_SIZE_Y + 1;

    let mut rng = rand::thread_rng();
    let a = NdArray::new_with_values([N, M], (0..N * M).map(|_| rng.gen_range(0., 10.)).collect())
        .unwrap();

    let b = NdArray::new_with_values([M, P], (0..M * P).map(|_| rng.gen_range(0., 10.)).collect())
        .unwrap();

    let mut c_gpu = NdArray::new(N * P);
    matmul_f32_impl([N, M, P], a.as_slice(), b.as_slice(), c_gpu.as_mut_slice()).unwrap();

    let mut c_cpu = NdArray::new(N * P);
    crate::ndarray::matrix::matmul_impl(
        [N, M, P],
        a.as_slice(),
        b.as_slice(),
        c_cpu.as_mut_slice(),
    )
    .unwrap();

    for (i, (a, b)) in c_gpu
        .as_slice()
        .iter()
        .zip(c_cpu.as_slice().iter())
        .enumerate()
    {
        assert!((a - b).abs() < 0.001, "{}: {} != {}", i, a, b)
    }
}

#[test]
fn test_larger_matrix() {
    const N: u32 = ROW_SPLIT_THRESHOLD * 2; // force combined GPU+CPU computation
    const M: u32 = 700;
    const P: u32 = LOCAL_SIZE_Y + 1;

    let mut rng = rand::thread_rng();
    let a = NdArray::new_with_values([N, M], (0..N * M).map(|_| rng.gen_range(0., 10.)).collect())
        .unwrap();

    let b = NdArray::new_with_values([M, P], (0..M * P).map(|_| rng.gen_range(0., 10.)).collect())
        .unwrap();

    let mut c_gpu = NdArray::new(N * P);
    matmul_f32_impl([N, M, P], a.as_slice(), b.as_slice(), c_gpu.as_mut_slice()).unwrap();

    let mut c_cpu = NdArray::new(N * P);
    crate::ndarray::matrix::matmul_impl(
        [N, M, P],
        a.as_slice(),
        b.as_slice(),
        c_cpu.as_mut_slice(),
    )
    .unwrap();

    for (i, (a, b)) in c_gpu
        .as_slice()
        .iter()
        .zip(c_cpu.as_slice().iter())
        .enumerate()
    {
        assert!((a - b).abs() < 0.01, "{}: {} != {}", i, a, b)
    }
}
