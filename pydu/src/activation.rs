use pyo3::{exceptions::PyValueError, prelude::*, wrap_pyfunction};

use crate::pyndarray::NdArrayD;

#[pyfunction]
pub fn relu(inp: PyRef<'_, NdArrayD>) -> NdArrayD {
    let res = du_core::activation::relu(&inp.inner);
    NdArrayD { inner: res }
}

#[pyfunction]
pub fn drelu_dz(inputs: PyRef<'_, NdArrayD>, dvalues: PyRef<'_, NdArrayD>) -> NdArrayD {
    let res = du_core::activation::drelu_dz(&inputs.inner, &dvalues.inner);
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
