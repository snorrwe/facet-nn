mod ndbool;
mod ndf64;
mod ndi64;

use facet_core::ndarray::NdArray;
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
        + Copy
        + Send
        + Sync;

    fn cast(&self) -> &NdArray<Self::T>;

    fn richcmp<F>(&self, other: PyRef<Self>, op: F) -> PyResult<NdArrayB>
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
        res.reshape(a.shape().clone());
        Ok(NdArrayB { inner: res })
    }

    fn add(lhs: PyRef<Self>, rhs: PyRef<Self>) -> PyResult<NdArray<Self::T>> {
        let lhs: &NdArray<Self::T> = lhs.cast();
        let rhs: &NdArray<Self::T> = rhs.cast();
        lhs.add(rhs)
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err)))
    }

    fn sub(lhs: PyRef<Self>, rhs: PyRef<Self>) -> PyResult<NdArray<Self::T>> {
        let lhs: &NdArray<Self::T> = lhs.cast();
        let rhs: &NdArray<Self::T> = rhs.cast();
        lhs.sub(rhs)
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err)))
    }

    fn mul(lhs: PyRef<Self>, rhs: PyRef<Self>) -> PyResult<NdArray<Self::T>> {
        let lhs: &NdArray<Self::T> = lhs.cast();
        let rhs: &NdArray<Self::T> = rhs.cast();
        lhs.mul(rhs)
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err)))
    }

    fn truediv(lhs: PyRef<Self>, rhs: PyRef<Self>) -> PyResult<NdArray<Self::T>> {
        let lhs: &NdArray<Self::T> = lhs.cast();
        let rhs: &NdArray<Self::T> = rhs.cast();
        lhs.div(rhs)
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err)))
    }

    fn pow(lhs: PyRef<Self>, rhs: Self::T) -> PyResult<NdArray<Self::T>>;
}

#[macro_export(internal_macros)]
macro_rules! impl_ndarray {
    ($ty: ty, $name: ident, $inner: ident, $mod: ident) => {
        mod $mod {
            use super::$name;
            use crate::pyndarray::PyNdIndex;
            use facet_core::ndarray::{column_iter::ColumnIter, shape::Shape, NdArray};
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
            pub struct ColIter {
                iter: ColumnIter<'static, $ty>,
                /// hold a reference to the original array to prevent the GC from collecting it
                arr: Option<Py<$name>>,
            }

            #[pyclass]
            pub struct ItemIter {
                iter: Box<dyn Iterator<Item = $ty> + Send + 'static>,
                /// hold a reference to the original array to prevent the GC from collecting it
                arr: Option<Py<$name>>,
            }

            #[pyproto]
            impl PyGCProtocol for ColIter {
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
            impl PyIterProtocol for ColIter {
                fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
                    this
                }

                fn __next__(mut this: PyRefMut<Self>) -> Option<Vec<$ty>> {
                    this.iter.next().map(|col| col.into())
                }
            }

            #[pyproto]
            impl PyGCProtocol for ItemIter {
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
            impl PyIterProtocol for ItemIter {
                fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
                    this
                }

                fn __next__(mut this: PyRefMut<Self>) -> Option<$ty> {
                    this.iter.next()
                }
            }
            #[pyproto]
            impl PyIterProtocol for $name {
                fn __iter__(this: PyRef<Self>) -> PyResult<ItemIter> {
                    let iter: Box<dyn Iterator<Item = _> + Send> =
                        Box::new(this.inner.iter().map(|x| *x));
                    // transmute the lifetime, we know this is safe because the iterator will hold
                    // a reference to this array, and Python is single threaded, so no mutations
                    // _should_ occur during iteration
                    let iter = unsafe { std::mem::transmute(iter) };
                    let iter = ItemIter {
                        iter,
                        arr: Some(this.into()),
                    };
                    Ok(iter)
                }
            }

            #[pymethods]
            impl $name {
                #[new]
                pub fn new(shape: &PyAny, values: Option<Vec<$ty>>) -> PyResult<Self> {
                    let shape = PyNdIndex::new(shape)?;
                    let inner = match values {
                        Some(v) => {
                            NdArray::new_with_values(shape.inner, v.into()).map_err(|err| {
                                PyValueError::new_err::<String>(format!("{}", err).into())
                            })?
                        }
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
                    this.inner.reshape(new_shape);
                    Ok(this)
                }

                pub fn get(&self, index: Vec<u32>) -> Option<$ty> {
                    self.inner.get(&index).cloned()
                }

                /// Treat this array as a flat array / vector and return the nth item
                pub fn flat_get(&self, index: usize) -> Option<$ty> {
                    self.inner.as_slice().get(index).cloned()
                }

                pub fn set(&mut self, index: Vec<u32>, value: $ty) {
                    if let Some(x) = self.inner.get_mut(&index) {
                        *x = value;
                    }
                }

                pub fn iter_rows<'py>(this: Py<Self>, py: Python<'py>) -> PyResult<Py<ColIter>> {
                    let s = this.borrow(py);
                    let it = s.inner.iter_rows();
                    // transmute the lifetime, we know this is safe because the iterator will hold
                    // a reference to this array, and Python is single threaded, so no mutations
                    // _should_ occur during iteration
                    let it = unsafe { std::mem::transmute(it) };
                    let it = ColIter {
                        iter: it,
                        arr: Some(this.clone()),
                    };
                    Py::new(py, it)
                }

                /// The values must have a length equal to the product of the dimensions!
                pub fn set_values(&mut self, values: Vec<$ty>) -> PyResult<()> {
                    self.inner
                        .set_slice(values.into())
                        .map(|_| ())
                        .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))
                }

                pub fn flip_mat_vertical(&self) -> PyResult<Self> {
                    let res = self
                        .inner
                        .flip_mat_vertical()
                        .map_err(|err| PyValueError::new_err::<String>(format!("{}", err)))?;
                    Ok(Self { inner: res })
                }

                pub fn flip_mat_horizontal(&self) -> PyResult<Self> {
                    let res = self
                        .inner
                        .flip_mat_horizontal()
                        .map_err(|err| PyValueError::new_err::<String>(format!("{}", err)))?;
                    Ok(Self { inner: res })
                }

                pub fn rotate_cw(&self) -> PyResult<Self> {
                    let res = self
                        .inner
                        .rotate_cw()
                        .map_err(|err| PyValueError::new_err::<String>(format!("{}", err)))?;
                    Ok(Self { inner: res })
                }

                pub fn to_string(&self) -> String {
                    self.inner.to_string()
                }

                pub fn transpose(&self) -> PyResult<Self> {
                    let res = self.inner.clone().transpose();
                    Ok(Self { inner: res })
                }

                /// Deep-copy this instance
                #[allow(clippy::should_implement_trait)] // this clone method is bridged to python
                pub fn clone(&self) -> Self {
                    Self {
                        inner: self.inner.clone(),
                    }
                }

                /// Call the given function with `(index, entry)` returning `None` leaves the entry
                /// unchanged, else replaces the given `entry` with the returned `value`
                pub fn replace_where(&mut self, cb: &PyAny) -> PyResult<()> {
                    for (i, entry) in self.inner.as_mut_slice().iter_mut().enumerate() {
                        let res = cb.call1((i, *entry))?;
                        match res.extract::<Option<$ty>>()? {
                            Some(x) => *entry = x,
                            None => {}
                        }
                    }
                    Ok(())
                }
            }

            #[pyproto]
            impl PyMappingProtocol for $name {
                fn __len__(&self) -> PyResult<usize> {
                    Ok(self.inner.shape().span())
                }

                fn __getitem__(&self, shape: &PyAny) -> PyResult<$ty> {
                    // TODO: not just single items.
                    // Shape could possibly hold fewer items than our shape, meaning they want
                    // vectors or matrices or sub-tensors returned...
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
