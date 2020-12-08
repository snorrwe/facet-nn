mod arrayimpl;
mod factory;

pub use self::arrayimpl::*;

use pyo3::{
    exceptions::PyNotImplementedError, exceptions::PyValueError, prelude::*, types::PyList,
    wrap_pyfunction,
};
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
    pub fn new(inp: &PyAny) -> PyResult<Self> {
        let shape = if let Ok(shape) = inp.extract::<Vec<u32>>() {
            shape
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

type Factory = fn(Python, Vec<u32>, &PyList) -> Result<Py<PyAny>, PyErr>;

#[pyfunction]
pub fn array(py: Python, inp: PyObject) -> PyResult<PyObject> {
    let mut dims = Vec::new();
    let inp: &PyList = inp.extract(py).or_else(|_| {
        inp.extract(py)
            .map_err(|err| {
                PyValueError::new_err(format!("Failed to convert input to a list {:?}", err))
            })
            .map(|f: f64| PyList::new(py, vec![f]))
    })?;
    let factory: Factory = {
        let mut inp = inp;
        loop {
            dims.push(u32::try_from(inp.len()).expect("expected dimensions to fit into 32 bits"));
            let i = inp.get_item(0);
            if let Ok(i) = i.downcast() {
                inp = i;
            } else if i.extract::<bool>().is_ok() {
                break factory::array_bool;
            } else if i.extract::<f64>().is_ok() {
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
