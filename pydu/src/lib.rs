pub mod activation;
pub mod io;
pub mod loss;
pub mod pyndarray;

use du_core::ndarray::{shape::Shape, NdArray};
use pyndarray::{NdArrayD, PyNdIndex};
use pyo3::{exceptions::PyValueError, prelude::*, wrap_pyfunction};

use std::convert::TryFrom;

/// Create a square matrix with `dims` columns and fill the main diagonal with 1's
#[pyfunction]
pub fn eye(dims: u32) -> NdArrayD {
    NdArrayD {
        inner: NdArray::diagonal(dims, 1.0),
    }
}

#[pyfunction]
pub fn zeros(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    let inp: PyNdIndex = inp
        .extract(py)
        .or_else(|_| PyNdIndex::new(inp.extract(py)?))?;

    let shape = Shape::from(inp.inner);

    let res = NdArray::new_default(shape);

    Ok(NdArrayD { inner: res })
}

/// Creates a square matrix where the diagonal holds the values of the input vector and the other
/// values are 0
#[pyfunction]
pub fn diagflat(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    let inp: Py<NdArrayD> = inp
        .extract(py)
        .or_else(|_| pyndarray::array(py, inp.extract(py)?)?.extract(py))?;
    let inp: &PyCell<NdArrayD> = inp.into_ref(py);
    let mut inp = inp.borrow_mut();
    let n = inp.inner.shape().span();
    let n = u32::try_from(n).map_err(|err| {
        PyValueError::new_err(format!("Failed to convert inp len to u32 {:?}", err))
    })?;
    inp.inner.reshape(n).unwrap();
    let inp = inp.inner.as_slice();
    let mut res = NdArray::new_default([n, n]);
    for i in 0..n {
        *res.get_mut(&[i, i]).unwrap() = inp[i as usize];
    }

    Ok(NdArrayD { inner: res })
}

#[pyfunction]
pub fn sum(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    let inp: Py<NdArrayD> = inp
        .extract(py)
        .or_else(|_| pyndarray::array(py, inp.extract(py)?)?.extract(py))?;

    let inp: &PyCell<NdArrayD> = inp.into_ref(py);
    let inp = inp.borrow();
    let res = inp
        .inner
        .iter_cols()
        .map(|x| x.iter().sum())
        .collect::<Vec<_>>();

    let shape = inp.shape();
    let shape = shape.as_slice();
    let res = if shape.len() > 0 {
        NdArray::new_with_values(&shape[..shape.len() - 1], res).unwrap()
    } else {
        // scalar
        NdArray::new_with_values(0, res).unwrap()
    };
    Ok(NdArrayD { inner: res })
}

/// Scrate a single-value nd-array
#[pyfunction]
pub fn scalar(s: f64) -> NdArrayD {
    NdArrayD {
        inner: NdArray::new_with_values(0, [s]).unwrap(),
    }
}

#[pymodule]
fn pydu(py: Python, m: &PyModule) -> PyResult<()> {
    pyndarray::setup_module(py, &m)?;
    activation::setup_module(py, &m)?;
    io::setup_module(py, &m)?;
    loss::setup_module(py, &m)?;

    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(diagflat, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(scalar, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;

    Ok(())
}
