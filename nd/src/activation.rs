use std::{convert::TryInto, f64::consts::E};

use pyo3::{prelude::*, wrap_pyfunction};

use crate::{ndarray::shape::Shape, ndarray::NdArray, pyndarray::NdArrayD};

#[pyfunction]
pub fn relu(inp: PyRef<'_, NdArrayD>) -> NdArrayD {
    let res = inp.inner.map(|v| v.max(0.0));
    NdArrayD { inner: res }
}

/// Inp is interpreted as a either a collection of vectors, applying softmax to each column or as a
/// single vector.
#[pyfunction]
pub fn softmax(inp: PyRef<'_, NdArrayD>) -> PyResult<NdArrayD> {
    // softmax of a scalar value is always 1.
    if matches!(inp.inner.shape, Shape::Scalar) {
        return Ok(NdArrayD {
            inner: NdArray::<f64> {
                shape: Shape::Scalar,
                values: [1.0].into(),
            },
        });
    }
    // else treat the input as a collection of vectors
    let inner: &NdArray<f64> = &inp.inner;
    let mut it = inner.values.iter().cloned();
    let first = it.next().expect("no value");
    let max: f64 = it.fold(first, |max, value| value.max(max));

    let expvalues = inner
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
    Ok(NdArrayD { inner: res })
}

pub fn setup_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    Ok(())
}
