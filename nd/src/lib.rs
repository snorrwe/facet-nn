pub mod ndarray;
pub mod pyndarray;

use pyndarray::NdArrayD;
use pyo3::prelude::*;

#[pymodule]
fn nd(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NdArrayD>()?;

    Ok(())
}
