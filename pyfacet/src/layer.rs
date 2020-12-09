//! Commonly used artificial neural network layer implementations
//!

pub mod dense_layer;

use pyo3::prelude::*;

pub fn setup_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<dense_layer::DenseLayer>()?;
    Ok(())
}
