pub mod activation;
pub mod io;
pub mod layer;
pub mod loss;
pub mod pyndarray;
use facet_core::rayon::iter::ParallelIterator;

use facet_core::ndarray::{shape::Shape, NdArray};
use pyndarray::{NdArrayD, NdArrayI, PyNdIndex};
use pyo3::{
    exceptions::{PyAssertionError, PyValueError},
    prelude::*,
    wrap_pyfunction,
};

use std::convert::TryFrom;

/// Create a square matrix with `dims` columns and fill the main diagonal with 1's
#[pyfunction]
pub fn eye(dims: u32) -> NdArrayD {
    NdArrayD {
        inner: NdArray::diagonal(dims, 1.0),
    }
}

fn pyobj_to_arrayd(py: Python, inp: PyObject) -> PyResult<Py<NdArrayD>> {
    let inp: Py<NdArrayD> = inp
        .extract(py)
        .or_else(|_| pyndarray::array(py, inp.extract(py)?)?.extract(py))
        .or_else(|_| inp.extract(py).and_then(|inp| Py::new(py, scalar(inp))))?;
    Ok(inp)
}

macro_rules! unwrap_obj {
    ($py: ident, $inp: ident) => {
        let $inp = pyobj_to_arrayd($py, $inp)?;
        let $inp = $inp.borrow($py);
    };

    (mut $py: ident, $inp: ident) => {
        let $inp = pyobj_to_arrayd($py, $inp)?;
        let mut $inp = $inp.borrow_mut($py);
    };
}

/// Collapses the last colun into a single index. The index of the largest item
#[pyfunction]
pub fn argmax(py: Python, inp: PyObject) -> PyResult<NdArrayI> {
    unwrap_obj!(py, inp);

    let res: Vec<i64> = inp
        .inner
        .par_iter_rows()
        .map(|row| {
            row.iter()
                .enumerate()
                .fold(0, |mi, (i, x)| if &row[mi] < x { i } else { mi }) as i64
        })
        .collect();

    let shape = inp.inner.shape();

    let mut res = NdArray::new_vector(res);
    res.reshape(shape.truncate());

    Ok(NdArrayI { inner: res })
}

/// Collapses the last colun into a single index. The index of the largest item
#[pyfunction]
pub fn argmin(py: Python, inp: PyObject) -> PyResult<NdArrayI> {
    unwrap_obj!(py, inp);

    let res: Vec<i64> = inp
        .inner
        .par_iter_rows()
        .map(|col| {
            col.iter()
                .enumerate()
                .fold(0, |mi, (i, x)| if &col[mi] > x { i } else { mi }) as i64
        })
        .collect();

    let shape = inp.inner.shape();

    let mut res = NdArray::new_vector(res);
    res.reshape(shape.truncate());

    Ok(NdArrayI { inner: res })
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

#[pyfunction]
pub fn ones(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    let inp: PyNdIndex = inp
        .extract(py)
        .or_else(|_| PyNdIndex::new(inp.extract(py)?))?;

    let shape = Shape::from(inp.inner);

    let values = (0..shape.span()).map(|_| 1.0).collect();
    let res = NdArray::new_with_values(shape, values).unwrap();

    Ok(NdArrayD { inner: res })
}

/// Creates a square matrix where the diagonal holds the values of the input vector and the other
/// values are 0
#[pyfunction]
pub fn diagflat(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    unwrap_obj!(mut py, inp);
    let n = inp.inner.shape().span();
    let n = u32::try_from(n).map_err(|err| {
        PyValueError::new_err(format!("Failed to convert inp len to u32 {:?}", err))
    })?;
    inp.inner.reshape(n);
    let inp = inp.inner.as_slice();
    let mut res = NdArray::new_default([n, n]);
    for i in 0..n {
        *res.get_mut(&[i, i]).unwrap() = inp[i as usize];
    }

    Ok(NdArrayD { inner: res })
}

#[pyfunction]
pub fn sum(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    let res = facet_core::sum(&inp.inner);
    Ok(NdArrayD { inner: res })
}

/// Scrate a single-value nd-array
#[pyfunction]
pub fn scalar(s: f32) -> NdArrayD {
    NdArrayD {
        inner: NdArray::new_with_values(0, (0..1).map(|_| s).collect()).unwrap(),
    }
}

#[pyfunction]
pub fn mean(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);
    facet_core::mean(&inp.inner)
        .map(|inner| NdArrayD { inner })
        .map_err(|err| PyValueError::new_err::<String>(format!("{}", err)))
}

#[pyfunction]
pub fn sqrt(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    let mut res = inp.clone();

    res.inner
        .as_mut_slice()
        .iter_mut()
        .for_each(|v| *v = v.sqrt());

    Ok(res)
}

#[pyfunction]
pub fn binomial(py: Python, n: u64, p: f32, size: Option<PyObject>) -> PyResult<NdArrayD> {
    use rand::prelude::*;

    let dist = rand_distr::Binomial::new(n, p.into()).map_err(|err| {
        PyAssertionError::new_err(format!(
            "Failed to create binomial distribution from the given arguments {}",
            err
        ))
    })?;

    let shape = size
        .and_then(|s| {
            let inp: PyNdIndex = s
                .extract(py)
                .or_else(|_| PyNdIndex::new(s.extract(py)?))
                .ok()?;

            let shape = Shape::from(inp.inner);
            Some(shape)
        })
        .unwrap_or_else(|| Shape::from(1));

    let len = shape.span();
    let mut res = NdArray::<f32>::new(shape);

    let mut rng = rand::thread_rng();
    for i in 0..len {
        let x = dist.sample(&mut rng);
        res.as_mut_slice()[i as usize] = x as f32;
    }

    let res = NdArrayD { inner: res };
    Ok(res)
}

#[pyfunction]
pub fn clip(py: Python, inp: PyObject, min: f32, max: f32) -> PyResult<NdArrayD> {
    // maybe throw a python exception?
    debug_assert!(min <= max);
    unwrap_obj!(py, inp);

    let mut res = inp.inner.clone();
    facet_core::clip(&mut res, min, max);

    let res = NdArrayD { inner: res };
    Ok(res)
}

#[pyfunction]
pub fn log(py: Python, inp: PyObject, base: Option<f32>) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    let base = base.unwrap_or(std::f32::consts::E);
    let res = inp.inner.as_slice().iter().map(|x| x.log(base)).collect();

    let inner = NdArray::new_with_values(inp.shape(), res).unwrap();

    Ok(NdArrayD { inner })
}

#[pyfunction]
pub fn std_squared(py: Python, inp: PyObject, mean: Option<PyObject>) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    let mean: Option<Py<NdArrayD>> = mean.and_then(|m| m.extract(py).ok());
    let mean = mean.as_ref().map(|m| m.borrow(py));

    let res =
        facet_core::std_squared(&inp.inner, mean.as_ref().map(|m| &(&*m).inner)).map_err(|e| {
            PyValueError::new_err(format!("Failed to perform std squared calculation {:?}", e))
        })?;

    Ok(NdArrayD { inner: res })
}

#[pyfunction]
pub fn std(py: Python, inp: PyObject, mean: Option<PyObject>) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    let mean: Option<Py<NdArrayD>> = mean.and_then(|m| m.extract(py).ok());
    let mean = mean.as_ref().map(|m| m.borrow(py));

    let res = facet_core::std(&inp.inner, mean.as_ref().map(|m| &(&*m).inner))
        .map_err(|e| PyValueError::new_err(format!("Failed to perform std calculation {:?}", e)))?;

    Ok(NdArrayD { inner: res })
}

#[pyfunction]
pub fn moving_average(py: Python, inp: PyObject, window: u64) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    facet_core::moving_average(&inp.inner, window as usize)
        .map_err(|err| PyValueError::new_err::<String>(format!("{}", err)))
        .map(|inner| NdArrayD { inner })
}

#[pyfunction]
pub fn veclen(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    let mut out = NdArray::new(0);

    facet_core::veclen(&inp.inner, &mut out);

    Ok(NdArrayD { inner: out })
}

#[pyfunction]
pub fn veclen_squared(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    let mut out = NdArray::new(0);
    facet_core::veclen_squared(&inp.inner, &mut out);

    Ok(NdArrayD { inner: out })
}

#[pyfunction]
pub fn normalize_vectors(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    let mut out = NdArray::new(0);
    facet_core::normalize_f32_vectors(&inp.inner, &mut out);

    Ok(NdArrayD { inner: out })
}

#[pyfunction]
pub fn fast_inverse_sqrt(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    let mut out = NdArray::new(0);
    facet_core::fast_inv_sqrt_f32(&inp.inner, &mut out);

    Ok(NdArrayD { inner: out })
}

#[pyfunction]
pub fn abs(py: Python, inp: PyObject) -> PyResult<NdArrayD> {
    unwrap_obj!(py, inp);

    let mut out = NdArray::new(inp.shape());

    for (y, x) in out
        .as_mut_slice()
        .iter_mut()
        .zip(inp.inner.as_slice().iter())
    {
        *y = x.abs();
    }

    Ok(NdArrayD { inner: out })
}

#[pymodule]
fn pyfacet(py: Python, m: &PyModule) -> PyResult<()> {
    pyndarray::setup_module(py, &m)?;
    activation::setup_module(py, &m)?;
    io::setup_module(py, &m)?;
    loss::setup_module(py, &m)?;
    layer::setup_module(py, &m)?;

    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(diagflat, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(scalar, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(binomial, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(clip, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(std_squared, m)?)?;
    m.add_function(wrap_pyfunction!(std, m)?)?;
    m.add_function(wrap_pyfunction!(moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(veclen, m)?)?;
    m.add_function(wrap_pyfunction!(veclen_squared, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(fast_inverse_sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;

    Ok(())
}
