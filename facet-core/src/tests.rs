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
