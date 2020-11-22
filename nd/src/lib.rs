pub mod activation;
pub mod ndarray;
pub mod pyndarray;

use pyndarray::NdArrayD;
use pyo3::prelude::*;

#[pymodule]
fn nd(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NdArrayD>()?;
    activation::setup_module(py, &m)?;

    Ok(())
}
