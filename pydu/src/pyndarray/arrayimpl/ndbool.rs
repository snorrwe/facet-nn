use du_core::ndarray::NdArray;
pub use ndarraybimpl::ColIter as ColIterB;
pub use ndarraybimpl::ItemIter as ItemIterB;
pub use ndarraybimpl::*;

use crate::impl_ndarray;
use crate::pyndarray::NdArrayD;

use pyo3::{basic::CompareOp, exceptions::PyNotImplementedError, prelude::*, PyObjectProtocol};

impl_ndarray!(bool, NdArrayB, inner, ndarraybimpl);

#[pyclass]
#[derive(Debug)]
pub struct NdArrayB {
    pub inner: NdArray<bool>,
}

#[pymethods]
impl NdArrayB {
    /// Return if all values are truthy
    pub fn all(&self) -> bool {
        self.inner.as_slice().iter().all(|x| *x)
    }

    /// Return if any value is truthy
    pub fn any(&self) -> bool {
        self.inner.as_slice().iter().any(|x| *x)
    }

    /// Convert self into float representation, where True becomes 1.0 and False becomes 0.0
    pub fn as_f64(&self) -> NdArrayD {
        let values = self
            .inner
            .as_slice()
            .iter()
            .map(|x| if *x { 1.0 } else { 0.0 })
            .collect();
        let res = NdArray::new_with_values(self.inner.shape().clone(), values).unwrap();
        NdArrayD { inner: res }
    }
}

#[pyproto]
impl<T> PyObjectProtocol for NdArrayB {
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
    fn __richcmp__(&'p self, other: PyRef<'p, Self>, op: CompareOp) -> PyResult<bool> {
        let op: fn(&bool, &bool) -> bool = match op {
            CompareOp::Gt | CompareOp::Ge | CompareOp::Le | CompareOp::Lt => {
                return Err(PyNotImplementedError::new_err(format!(
                    "Op: {:?} is not implemented for boolean arrays",
                    op
                )))
            }
            CompareOp::Eq => |a, b| a == b,
            CompareOp::Ne => |a, b| a != b,
        };

        Ok(self
            .inner
            .as_slice()
            .iter()
            .zip(other.inner.as_slice().iter())
            .all(move |(a, b)| op(a, b)))
    }
}
