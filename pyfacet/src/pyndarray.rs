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
pub fn array(py: Python, shape: PyObject) -> PyResult<PyObject> {
    let mut dims = Vec::new();
    let shape: &PyList = shape.extract(py).or_else(|_| {
        shape.extract(py)
            .map_err(|err| {
                PyValueError::new_err(format!("Failed to convert input to a list {:?}", err))
            })
            .map(|f: f32| PyList::new(py, vec![f]))
    })?;
    let factory: Factory = {
        let mut shape = shape;
        loop {
            dims.push(u32::try_from(shape.len()).expect("expected dimensions to fit into 32 bits"));
            let i = shape.get_item(0);
            if let Ok(i) = i.downcast() {
                shape = i;
            } else if i.extract::<bool>().is_ok() {
                break factory::array_bool;
            } else if i.extract::<f32>().is_ok() {
                break factory::array_f32;
            } else {
                return Err(PyNotImplementedError::new_err(format!(
                    "Value with unexpected type: {:?}",
                    i
                )));
            }
        }
    };

    factory(py, dims, shape)
}
