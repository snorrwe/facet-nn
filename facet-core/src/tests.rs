use crate::prelude::*;

#[test]
fn test_moving_average_matrix() {
    let values = NdArray::new_with_values(
        [2, 12],
        smallvec![
            9.0, 8.0, 9.0, 12.0, 9., 12., 11., 7., 13., 9., 11., 10., //
            9.0, 8.0, 9.0, 12.0, 9., 12., 11., 7., 13., 9., 11., 10.
        ],
    )
    .unwrap();
    let values = values.transpose();
    let ma = crate::moving_average(&values, 3).unwrap();
    let ma = ma.transpose();

    assert_eq!(
        ma.as_slice(),
        &[
            8.666666666666666,
            9.666666666666666,
            10.0,
            11.0,
            10.666666666666666,
            10.0,
            10.333333333333334,
            9.666666666666666,
            11.0,
            10.0,
            // row 2
            8.666666666666666,
            9.666666666666666,
            10.0,
            11.0,
            10.666666666666666,
            10.0,
            10.333333333333334,
            9.666666666666666,
            11.0,
            10.0
        ]
    );
}

#[test]
fn test_fast_inv_sqrt_accuracy() {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let inp: Data<f32> = (0..10000).map(|_| rng.gen_range(0.0, 10.0)).collect();

    let inp = NdArray::new_with_values([2000, 5], inp).unwrap();
    let mut out = NdArray::new_default(inp.shape().clone());

    crate::fast_inv_sqrt_f32(&inp, &mut out).unwrap();

    for (y, x) in out.as_slice().iter().zip(inp.as_slice().iter()) {
        let exp = 1.0 / x.sqrt();

        let diff = y - exp;
        let error = diff.abs() / y;
        assert!(
            error < 0.01,
            "y: {}, exp: {}, diff: {} error: {}",
            y,
            exp,
            diff,
            error
        );
    }
}

#[test]
fn test_f32_vec_norm_accuracy() {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let inp: Data<f32> = (0..10000).map(|_| rng.gen_range(0.0, 10.0)).collect();

    let inp = NdArray::new_with_values([2000, 5], inp).unwrap();
    let mut out = NdArray::new_default(inp.shape().clone());

    crate::normalize_f32_vectors(&inp, &mut out).unwrap();

    for y in out.iter_cols() {
        let len = y.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(len <= 1.0, "{}", len);
        let err = (len - 1.0).abs();
        assert!(err < 0.002, "len {} err {}", len, err);
    }
}
