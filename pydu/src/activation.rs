use pyo3::{exceptions::PyValueError, prelude::*, wrap_pyfunction};

use crate::pyndarray::NdArrayD;

#[pyfunction]
pub fn relu(inp: PyRef<'_, NdArrayD>) -> NdArrayD {
    let res = inp.inner.map(|v| v.max(0.0));
    NdArrayD { inner: res }
}

#[pyfunction]
pub fn drelu_dz(z: PyRef<'_, NdArrayD>) -> NdArrayD {
    let res = z.inner.map(|v| if *v > 0.0 { 1.0 } else { 0.0 });
    NdArrayD { inner: res }
}

/// Inp is interpreted as a either a collection of vectors, applying softmax to each column or as a
/// single vector.
///
/// Scalars will always return 1
#[pyfunction]
pub fn softmax(inp: PyRef<'_, NdArrayD>) -> PyResult<NdArrayD> {
    du_core::activation::softmax(&inp.inner)
        .map_err(|err| PyValueError::new_err(format!("Failed to perform softmax {}", err)))
        .map(|inner| NdArrayD { inner })
}

pub fn setup_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(drelu_dz, m)?)?;
    Ok(())
}
