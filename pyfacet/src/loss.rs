use crate::pyndarray::NdArrayD;
use pyo3::{exceptions::PyValueError, prelude::*, wrap_pyfunction};

/// The author recommends running `softmax` on the output before calling this function
#[pyfunction]
pub fn categorical_cross_entropy(predictions: &NdArrayD, targets: &NdArrayD) -> PyResult<NdArrayD> {
    facet_core::loss::categorical_cross_entropy(&predictions.inner, &targets.inner)
        .map_err(|err| PyValueError::new_err(format!("Failed to perform CCE: {}", err)))
        .map(|inner| NdArrayD { inner })
}

pub fn setup_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(categorical_cross_entropy, m)?)?;
    Ok(())
}
