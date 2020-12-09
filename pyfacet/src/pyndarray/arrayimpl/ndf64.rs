use facet_core::ndarray::NdArray;
pub use ndarraydimpl::ColIter as ColIterD;
pub use ndarraydimpl::ItemIter as ItemIterD;
pub use ndarraydimpl::*;

use crate::impl_ndarray;

use pyo3::{
    basic::CompareOp,
    exceptions::{PyNotImplementedError, PyValueError},
    prelude::*,
    PyNumberProtocol, PyObjectProtocol,
};

use super::AsNumArray;
use super::NdArrayB;

impl_ndarray!(f64, NdArrayD, inner, ndarraydimpl);

#[pyclass]
#[derive(Debug, Clone)]
pub struct NdArrayD {
    pub inner: NdArray<f64>,
}

#[pyproto]
impl<T> PyNumberProtocol for NdArrayD {
    fn __add__(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<Self> {
        <Self as AsNumArray>::add(lhs, rhs).map(|inner| Self { inner })
    }

    fn __sub__(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<Self> {
        <Self as AsNumArray>::sub(lhs, rhs).map(|inner| Self { inner })
    }

    fn __mul__(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<Self> {
        <Self as AsNumArray>::mul(lhs, rhs).map(|inner| Self { inner })
    }

    fn __truediv__(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<Self> {
        <Self as AsNumArray>::truediv(lhs, rhs).map(|inner| Self { inner })
    }

    fn __pow__(lhs: PyRef<'p, Self>, rhs: f64, _modulo: Option<f64>) -> PyResult<Self> {
        <Self as AsNumArray>::pow(lhs, rhs).map(|inner| Self { inner })
    }
}

#[pyproto]
impl PyObjectProtocol for NdArrayD {
    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "NdArray of f64, shape: {:?}, data:\n{}",
            self.inner.shape(),
            self.to_string()
        )
    }

    fn __bool__(&'p self) -> PyResult<bool> {
        Err(PyNotImplementedError::new_err::<String>(
            "Array to bool conversion is ambigous! Use .any or .all".to_string(),
        ))
    }

    /// Returns an NdArray where each element is 1 if true 0 if false for the given pair of
    /// elements.
    fn __richcmp__(&'p self, other: PyRef<'p, Self>, op: CompareOp) -> PyResult<NdArrayB> {
        let op: fn(&f64, &f64) -> bool = match op {
            CompareOp::Lt => |a, b| a < b,
            CompareOp::Le => |a, b| a <= b,
            CompareOp::Eq => |a, b| (a - b).abs() < std::f64::EPSILON,
            CompareOp::Ne => |a, b| (a - b).abs() >= std::f64::EPSILON,
            CompareOp::Gt => |a, b| a > b,
            CompareOp::Ge => |a, b| a >= b,
        };
        self.richcmp(other, op)
    }
}

impl AsNumArray for NdArrayD {
    type T = f64;

    fn cast(&self) -> &NdArray<Self::T> {
        &self.inner
    }

    fn pow(lhs: PyRef<Self>, rhs: Self::T) -> PyResult<NdArray<f64>> {
        let lhs: &NdArray<Self::T> = lhs.cast();
        let res = lhs.map(|x| x.powf(rhs));
        Ok(res)
    }
}

#[pymethods]
impl NdArrayD {
    pub fn matmul(
        this: PyRef<Self>,
        other: &Self,
        mut out: Option<PyRefMut<Self>>,
    ) -> PyResult<PyObject> {
        let mut _out = NdArray::new(0);
        let outref = out.as_mut().map(|m| &mut m.inner).unwrap_or(&mut _out);
        this.inner
            .matmul(&other.inner, outref)
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err)))?;
        let py = this.py();
        let out = out.map(|m| m.into_py(py)).unwrap_or_else(|| {
            let res = NdArrayD { inner: _out };
            let res = Py::new(py, res).unwrap();
            res.into_py(py)
        });
        Ok(out)
    }

    pub fn clip(mut this: PyRefMut<Self>, min: f64, max: f64) -> PyResult<PyRefMut<Self>> {
        this.inner
            .as_mut_slice()
            .iter_mut()
            .for_each(|v| *v = v.max(min).min(max));

        Ok(this)
    }
}
