use std::{convert::TryInto, f64::consts::E};

use crate::{ndarray::shape::Shape, ndarray::NdArray, DuResult};

/// ReLU
pub fn relu(inp: &NdArray<f64>) -> NdArray<f64> {
    inp.map(|v| v.max(0.0))
}

/// derivative of the ReLU function
pub fn drelu_dz(z: &NdArray<f64>) -> NdArray<f64> {
    z.map(|v| if *v > 0.0 { 1.0 } else { 0.0 })
}

/// Inp is interpreted as a either a collection of vectors, applying softmax to each column or as a
/// single vector.
///
/// Scalars will always return 1
pub fn softmax(inp: &NdArray<f64>) -> DuResult<NdArray<f64>> {
    // softmax of a scalar value is always 1.
    if matches!(inp.shape, Shape::Scalar) {
        return Ok(NdArray::<f64> {
            shape: Shape::Scalar,
            values: [1.0].into(),
        });
    }
    // else treat the input as a collection of vectors
    let mut it = inp.values.iter().cloned();
    let first = it.next().expect("no value");
    let max: f64 = it.fold(first, |max, value| value.max(max));

    let expvalues = inp
        .sub(&NdArray::from(max))
        .expect("Failed to sub max from the input")
        .map(|v: &f64| E.powf(*v));

    let mut norm_base: NdArray<f64> = expvalues
        .iter_cols()
        .map(|col| col.iter().cloned().sum())
        .collect();

    debug_assert_eq!(
        norm_base.shape,
        Shape::Vector(
            (expvalues.shape.span() / expvalues.shape.last().unwrap() as usize)
                .try_into()
                .expect("failed to convert vector len to u32")
        ),
        "internal error when producing norm_base"
    );

    norm_base
        .reshape(Shape::Matrix(norm_base.shape.last().unwrap(), 1))
        .unwrap();

    let mut res = expvalues;
    for (norm, col) in norm_base.iter_cols().zip(res.iter_cols_mut()) {
        debug_assert_eq!(norm.len(), 1);
        col.iter_mut().for_each(|v| *v /= norm[0]);
    }
    Ok(res)
}
