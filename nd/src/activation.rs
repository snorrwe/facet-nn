use std::f64::EPSILON;

use pyo3::{prelude::*, wrap_pyfunction};

use crate::{ndarray::shape::Shape, ndarray::NdArray, pyndarray::NdArrayD};

#[pyfunction]
pub fn relu(inp: PyRef<'_, NdArrayD>) -> NdArrayD {
    let res = inp.inner.map(|v| v.max(0.0));
    NdArrayD { inner: res }
}

#[pyfunction]
pub fn softmax(inp: PyRef<'_, NdArrayD>) -> PyResult<NdArrayD> {
    todo!()
}

#[pymodule]
pub fn activation(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    Ok(())
}
