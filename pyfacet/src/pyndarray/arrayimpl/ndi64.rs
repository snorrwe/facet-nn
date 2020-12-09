use crate::impl_ndarray;
use facet_core::ndarray::NdArray;
pub use implmod::ColIter as ColIterI;
pub use implmod::ItemIter as ItemIterI;
pub use implmod::*;

use pyo3::{
    basic::CompareOp, exceptions::PyNotImplementedError, prelude::*, PyNumberProtocol,
    PyObjectProtocol,
};

use super::{AsNumArray, NdArrayB, NdArrayD};

impl_ndarray!(i64, NdArrayI, inner, implmod);

/// Index array
#[pyclass]
#[derive(Debug)]
pub struct NdArrayI {
    pub inner: NdArray<i64>,
}

#[pyproto]
impl<T> PyNumberProtocol for NdArrayI {
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
impl<T> PyObjectProtocol for NdArrayI {
    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "NdArray of i64, shape: {:?}, data:\n{}",
            self.inner.shape(),
            self.to_string()
        )
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyNotImplementedError::new_err::<String>(
            "Array to bool conversion is ambigous! Use .any or .all".to_string(),
        ))
    }

    /// Returns an NdArray where each element is 1 if true 0 if false for the given pair of
    /// elements.
    fn __richcmp__(&self, other: PyRef<Self>, op: CompareOp) -> PyResult<NdArrayB> {
        let op: fn(&i64, &i64) -> bool = match op {
            CompareOp::Lt => |a, b| a < b,
            CompareOp::Le => |a, b| a <= b,
            CompareOp::Eq => |a, b| a == b,
            CompareOp::Ne => |a, b| a != b,
            CompareOp::Gt => |a, b| a > b,
            CompareOp::Ge => |a, b| a >= b,
        };
        self.richcmp(other, op)
    }
}

impl AsNumArray for NdArrayI {
    type T = i64;

    fn cast(&self) -> &NdArray<Self::T> {
        &self.inner
    }

    fn pow(lhs: PyRef<Self>, rhs: Self::T) -> PyResult<NdArray<Self::T>> {
        let lhs: &NdArray<Self::T> = lhs.cast();
        let res = lhs.map(|x| x.pow(rhs as u32));
        Ok(res)
    }
}

#[pymethods]
impl NdArrayI {
    /// Convert self into float representation
    pub fn as_f64(&self) -> NdArrayD {
        let values = self.inner.as_slice().iter().map(|x| *x as f64).collect();
        let res = NdArray::new_with_values(self.inner.shape().clone(), values).unwrap();
        NdArrayD { inner: res }
    }
}
