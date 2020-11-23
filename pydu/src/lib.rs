pub mod activation;
pub mod pyndarray;
pub mod io;
pub mod loss;

use pyo3::prelude::*;

#[pymodule]
fn du(py: Python, m: &PyModule) -> PyResult<()> {
    pyndarray::setup_module(py, &m)?;
    activation::setup_module(py, &m)?;
    io::setup_module(py, &m)?;
    loss::setup_module(py, &m)?;

    Ok(())
}
