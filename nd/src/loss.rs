use std::f64::consts::E;

use crate::{ndarray::NdArray, pyndarray::NdArrayD};
use pyo3::{exceptions::PyValueError, prelude::*, wrap_pyfunction};

/// Note that 0 inputs will produce `NaN` output.
///
/// The author recommends running `softmax` on the output before calling this function
#[pyfunction]
pub fn categorical_cross_entropy(predictions: &NdArrayD, targets: &NdArrayD) -> PyResult<NdArrayD> {
    if predictions.inner.shape != targets.inner.shape {
        return Err(PyValueError::new_err(format!(
            "function requires matching shapes, but got: {:?} and {:?}",
            predictions.inner.shape, targets.inner.shape,
        )));
    }

    let mut out = Vec::with_capacity(
        predictions.inner.shape.span() / predictions.inner.shape.last().unwrap_or(1) as usize,
    );
    for (x, y) in predictions.inner.iter_cols().zip(targets.inner.iter_cols()) {
        let loss: f64 = x
            .iter()
            .cloned()
            .zip(y.iter().cloned())
            .map(|(x, y)| -> f64 { x.log(E) * y })
            .sum();
        // loss is always < 0 so let's flip it...
        out.push(-loss);
    }

    let out = NdArray::new_with_values(out.len() as u64, out.into_boxed_slice()).unwrap();
    Ok(NdArrayD { inner: out })
}

pub fn setup_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(categorical_cross_entropy, m)?)?;
    Ok(())
}
