mod arrayimpl;
mod factory;

pub use self::arrayimpl::*;

use pyo3::{exceptions::PyNotImplementedError, prelude::*, types::PyList, wrap_pyfunction};
use std::convert::TryFrom;

pub fn setup_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_class::<NdArrayD>()?;
    m.add_class::<NdArrayB>()?;
    m.add_class::<NdArrayI>()?;
    m.add_class::<PyNdIndex>()?;
    Ok(())
}

#[pyclass]
#[derive(Clone)]
pub struct PyNdIndex {
    pub inner: Vec<u32>,
}

#[pymethods]
impl PyNdIndex {
    #[new]
    fn new(inp: &PyAny) -> PyResult<Self> {
        let shape = if let Ok(lst) = inp.extract::<Vec<u32>>() {
            lst.into()
        } else if let Ok(n) = inp.extract::<u32>() {
            if n == 0 {
                vec![0]
            } else {
                vec![n]
            }
        } else {
            todo!()
        };
        Ok(Self { inner: shape })
    }
}

#[pyfunction]
pub fn array(py: Python, inp: &PyList) -> PyResult<PyObject> {
    let mut dims = Vec::new();
    let factory: fn(Python, Vec<u32>, &PyList) -> Result<Py<PyAny>, PyErr> = {
        let mut inp = inp;
        loop {
            dims.push(u32::try_from(inp.len()).expect("expected dimensions to fit into 32 bits"));
            let i = inp.get_item(0);
            if let Ok(i) = i.downcast() {
                inp = i;
            } else if let Ok(_) = i.extract::<bool>() {
                break factory::array_bool;
            } else if let Ok(_) = i.extract::<f64>() {
                break factory::array_f64;
            } else {
                return Err(PyNotImplementedError::new_err(format!(
                    "Value with unexpected type: {:?}",
                    i
                )));
            }
        }
    };

    factory(py, dims, inp)
}
