pub mod activation;
pub mod io;
pub mod loss;
pub mod pyndarray;

use du_core::ndarray::NdArray;
use pyndarray::NdArrayD;
use pyo3::{exceptions::PyValueError, prelude::*, wrap_pyfunction};

use std::convert::TryFrom;

/// Create a square matrix with `dims` columns and fill the main diagonal with 1's
#[pyfunction]
pub fn eye(dims: u32) -> NdArrayD {
    NdArrayD {
        inner: NdArray::diagonal(dims, 1.0),
    }
}

/// Creates a squre matrix where the diagonal holds the values of the input vector and the other
/// values are 0
#[pyfunction]
pub fn diagflat(inp: Vec<f64>) -> PyResult<NdArrayD> {
    let n = u32::try_from(inp.len())
        .map_err(|err| PyValueError::new_err(format!("Failed to convert inp to u32 {:?}", err)))?;
    let mut res = NdArray::new_default([n, n]);
    for i in 0..n {
        *res.get_mut(&[i, i]).unwrap() = inp[i as usize];
    }

    Ok(NdArrayD { inner: res })
}

#[pymodule]
fn pydu(py: Python, m: &PyModule) -> PyResult<()> {
    pyndarray::setup_module(py, &m)?;
    activation::setup_module(py, &m)?;
    io::setup_module(py, &m)?;
    loss::setup_module(py, &m)?;

    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(diagflat, m)?)?;

    Ok(())
}
