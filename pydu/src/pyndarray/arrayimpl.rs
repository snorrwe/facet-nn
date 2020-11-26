mod ndbool;
mod ndf64;
mod ndi64;

use du_core::ndarray::NdArray;
pub use ndbool::*;
pub use ndf64::*;
pub use ndi64::*;

use pyo3::{exceptions::PyValueError, prelude::*, PyClass};
use std::{
    ops::Add, ops::AddAssign, ops::Div, ops::DivAssign, ops::Mul, ops::MulAssign, ops::Sub,
    ops::SubAssign,
};

trait AsNumArray: PyClass {
    type T: Add<Self::T, Output = Self::T>
        + AddAssign
        + Sub<Self::T, Output = Self::T>
        + SubAssign
        + Mul<Self::T, Output = Self::T>
        + MulAssign
        + Div<Self::T, Output = Self::T>
        + DivAssign
        + Default
        + Copy;

    fn cast(&self) -> &NdArray<Self::T>;

    fn richcmp<'p, F>(&'p self, other: PyRef<'p, Self>, op: F) -> PyResult<NdArrayB>
    where
        F: Fn(&Self::T, &Self::T) -> bool,
    {
        // TODO: check shape and raise error on uncomparable shapes
        // TODO: support different shapes e.g. compare each element to a scalar
        let a = self.cast();
        let b = other.cast();
        let values: Vec<_> = a
            .as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .map(|(a, b)| op(a, b))
            .collect();
        let mut res = NdArray::<bool>::new_vector(values);
        res.reshape(a.shape().clone())
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))?;
        Ok(NdArrayB { inner: res })
    }

    fn matmul<'p>(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<NdArray<Self::T>> {
        NdArray::<Self::T>::matmul(lhs.cast(), rhs.cast())
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))
    }

    fn add<'p>(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<NdArray<Self::T>> {
        let lhs: &NdArray<Self::T> = lhs.cast();
        let rhs: &NdArray<Self::T> = rhs.cast();
        lhs.add(rhs)
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))
    }

    fn sub<'p>(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<NdArray<Self::T>> {
        let lhs: &NdArray<Self::T> = lhs.cast();
        let rhs: &NdArray<Self::T> = rhs.cast();
        lhs.sub(rhs)
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))
    }

    fn mul<'p>(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<NdArray<Self::T>> {
        let lhs: &NdArray<Self::T> = lhs.cast();
        let rhs: &NdArray<Self::T> = rhs.cast();
        lhs.mul(rhs)
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))
    }

    fn truediv<'p>(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<NdArray<Self::T>> {
        let lhs: &NdArray<Self::T> = lhs.cast();
        let rhs: &NdArray<Self::T> = rhs.cast();
        lhs.div(rhs)
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))
    }

    fn pow<'p>(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<NdArray<Self::T>>;
}

#[macro_export(internal_macros)]
macro_rules! impl_ndarray {
    ($ty: ty, $name: ident, $inner: ident, $itname: ident, $mod: ident) => {
        mod $mod {
            use super::$name;
            use crate::pyndarray::PyNdIndex;
            use du_core::ndarray::{column_iter::ColumnIter, shape::Shape, NdArray};
            use pyo3::{
                exceptions::{PyIndexError, PyValueError},
                prelude::*,
                PyGCProtocol, PyIterProtocol, PyMappingProtocol,
            };

            impl From<NdArray<$ty>> for $name {
                fn from(inner: NdArray<$ty>) -> Self {
                    Self { inner }
                }
            }

            #[pyclass]
            pub struct $itname {
                iter: ColumnIter<'static, $ty>,
                /// hold a reference to the original array to prevent the GC from collecting it
                arr: Option<Py<$name>>,
            }

            #[pyproto]
            impl PyGCProtocol for $itname {
                fn __traverse__(
                    &'p self,
                    visit: pyo3::PyVisit,
                ) -> Result<(), pyo3::PyTraverseError> {
                    visit.call(&self.arr)
                }

                fn __clear__(&'p mut self) {
                    self.arr = None
                }
            }

            #[pyproto]
            impl PyIterProtocol for $itname {
                fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
                    slf
                }

                fn __next__(mut slf: PyRefMut<Self>) -> Option<Vec<$ty>> {
                    slf.iter.next().map(|col| col.into())
                }
            }

            #[pymethods]
            impl $name {
                #[new]
                pub fn new(shape: &PyAny, values: Option<Vec<$ty>>) -> PyResult<Self> {
                    let shape = PyNdIndex::new(shape)?;
                    let inner = match values {
                        Some(v) => NdArray::new_with_values(shape.inner, v.into_boxed_slice())
                            .map_err(|err| {
                                PyValueError::new_err::<String>(format!("{}", err).into())
                            })?,
                        None => NdArray::new(shape.inner),
                    };
                    Ok(Self { inner })
                }

                #[getter]
                pub fn shape(&self) -> Vec<u32> {
                    match self.inner.shape() {
                        Shape::Scalar(_) => vec![],
                        Shape::Vector([n]) => vec![*n],
                        Shape::Matrix([n, m]) => vec![*n, *m],
                        Shape::Tensor(s) => s.clone().into_vec(),
                    }
                }

                #[getter]
                #[allow(non_snake_case)]
                pub fn T(&self) -> PyResult<Self> {
                    self.transpose()
                }

                pub fn reshape(
                    mut this: PyRefMut<Self>,
                    new_shape: Vec<u32>,
                ) -> PyResult<PyRefMut<Self>> {
                    this.inner.reshape(Shape::from(new_shape)).map_err(|err| {
                        PyValueError::new_err::<String>(format!("{}", err).into())
                    })?;
                    Ok(this)
                }

                pub fn get(&self, index: Vec<u32>) -> Option<$ty> {
                    self.inner.get(&index).cloned()
                }

                pub fn set(&mut self, index: Vec<u32>, value: $ty) {
                    if let Some(x) = self.inner.get_mut(&index) {
                        *x = value;
                    }
                }

                pub fn iter_cols<'py>(slf: Py<Self>, py: Python<'py>) -> PyResult<Py<$itname>> {
                    let s = slf.borrow(py);
                    let it = s.inner.iter_cols();
                    let it = unsafe { std::mem::transmute(it) };
                    let it = $itname {
                        iter: it,
                        arr: Some(slf.clone()),
                    };
                    Py::new(py, it)
                }

                /// The values must have a length equal to the product of the dimensions!
                pub fn set_values(&mut self, values: Vec<$ty>) -> PyResult<()> {
                    self.inner
                        .set_slice(values.into_boxed_slice())
                        .map(|_| ())
                        .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))
                }

                pub fn to_string(&self) -> String {
                    self.inner.to_string()
                }

                pub fn transpose(&self) -> PyResult<Self> {
                    let res = self.inner.clone().transpose();
                    Ok(Self { inner: res })
                }

                /// Deep-copy this instance
                pub fn clone(&self) -> Self {
                    Self {
                        inner: self.inner.clone(),
                    }
                }
            }

            #[pyproto]
            impl PyMappingProtocol for $name {
                fn __len__(&self) -> PyResult<usize> {
                    Ok(self.inner.shape().span())
                }

                fn __getitem__(&self, shape: &PyAny) -> PyResult<$ty> {
                    let shape = PyNdIndex::new(shape)?;
                    self.inner
                        .get(&shape.inner[..])
                        .ok_or_else(|| {
                            PyIndexError::new_err(format!(
                                "can't find item at index {:?}",
                                shape.inner
                            ))
                        })
                        .map(|x| x.clone())
                }
            }
        }
    };
}
