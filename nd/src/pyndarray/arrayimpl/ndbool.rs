use crate::pyndarray::NdArrayD;
use crate::{impl_ndarray, ndarray::NdArray};

use pyo3::{basic::CompareOp, exceptions::PyNotImplementedError, prelude::*, PyObjectProtocol};

impl_ndarray!(bool, NdArrayB, NdArrayBColIter, ndarraybimpl);
pub use ndarraybimpl::*;

#[pymethods]
impl NdArrayB {
    /// Return if all values are truthy
    pub fn all(&self) -> bool {
        self.inner.values.iter().all(|x| *x)
    }

    /// Return if any value is truthy
    pub fn any(&self) -> bool {
        self.inner.values.iter().any(|x| *x)
    }

    /// Convert self into float representation, where True becomes 1.0 and False becomes 0.0
    pub fn as_f64(&self) -> NdArrayD {
        let values: Vec<f64> = self
            .inner
            .values
            .iter()
            .map(|x| if *x { 1.0 } else { 0.0 })
            .collect();
        let res = NdArray::new_with_values(self.inner.shape.clone(), values).unwrap();
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
            .values
            .iter()
            .zip(other.inner.values.iter())
            .all(move |(a, b)| op(a, b)))
    }
}
