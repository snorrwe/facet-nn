use crate::impl_ndarray;

use crate::ndarray::NdArray;
use pyo3::{
    basic::CompareOp,
    exceptions::{PyNotImplementedError, PyValueError},
    prelude::*,
    PyNumberProtocol, PyObjectProtocol,
};

use super::AsNumArray;
use super::NdArrayB;

impl_ndarray!(f64, NdArrayD, NdArrayDColIter, ndarraydimpl);
pub use ndarraydimpl::*;

#[pyproto]
impl<T> PyNumberProtocol for NdArrayD {
    fn __matmul__(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<Self> {
        lhs.matmul(&*rhs)
    }

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
}

#[pyproto]
impl<T> PyObjectProtocol for NdArrayD {
    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "NdArray of f64, shape: {:?}, data:\n{}",
            self.inner.shape,
            self.to_string()
        )
    }

    fn __bool__(&'p self) -> PyResult<bool> {
        Err(PyNotImplementedError::new_err::<String>(
            format!("Array to bool conversion is ambigous! Use .any or .all").into(),
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
}

#[pymethods]
impl NdArrayD {
    pub fn matmul(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .matmul(&other.inner)
            .map(|inner| Self { inner })
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))
    }
}
