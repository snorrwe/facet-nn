pub mod activation;
pub mod io;
pub mod ndarray;
pub mod pyndarray;

use pyo3::prelude::*;

#[pymodule]
fn nd(py: Python, m: &PyModule) -> PyResult<()> {
    pyndarray::setup_module(py, &m)?;
    activation::setup_module(py, &m)?;
    io::setup_module(py, &m)?;

    Ok(())
}
