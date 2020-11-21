pub mod activation;
pub mod ndarray;
pub mod pyndarray;

use pyndarray::NdArrayD;
use pyo3::prelude::*;

#[pymodule]
fn nd(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NdArrayD>()?;

    let acti = PyModule::new(py, "activation")?;
    activation::activation(py, acti)?;
    m.add_submodule(acti)?;

    Ok(())
}
