mod arrayimpl;
mod factory;

pub use self::arrayimpl::*;

use pyo3::{prelude::*, types::PyList, wrap_pyfunction};
use std::convert::TryFrom;

pub fn setup_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_class::<NdArrayD>()?;
    Ok(())
}

#[pyfunction]
pub fn array(py: Python, inp: &PyList) -> PyResult<PyObject> {
    let mut dims = Vec::new();
    let factory = {
        let mut inp = inp;
        loop {
            dims.push(u32::try_from(inp.len()).expect("expected dimensions to fit into 32 bits"));
            let i = inp.get_item(0);
            if let Ok(i) = i.downcast() {
                inp = i;
            } else if let Ok(_) = i.extract::<f64>() {
                break factory::array_f64;
            } else {
                panic!("Value with unexpected type: {:?}", i);
            }
        }
    };

    factory(py, dims, inp)
}
