use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};

use crate::ndarray::NdArray;

use super::NdArrayD;

pub fn array_f64(py: Python, dims: Vec<u32>, inp: &PyList) -> PyResult<PyObject> {
    let mut values: Vec<f64> = Vec::new(); // TODO: reserve
    flatten(inp, &mut values)?;
    let arr = NdArray::<f64>::new_with_values(dims, values.into_boxed_slice()).map_err(|err| {
        PyValueError::new_err::<String>(format!("Failed to create nd-array of f64: {}", err).into())
    })?;
    let res = NdArrayD { inner: arr };
    let res = Py::new(py, res)?;

    // cast result as Any
    let res = unsafe { Py::from_owned_ptr(py, res.into_ptr()) };
    Ok(res)
}

fn flatten<'a, T: Clone + FromPyObject<'a>>(inp: &'a PyList, out: &mut Vec<T>) -> PyResult<()> {
    for val in inp.iter() {
        if let Ok(l) = val.downcast() {
            flatten(l, out)?;
        } else {
            out.push(val.extract()?);
        }
    }
    Ok(())
}
