use pyo3::{
    basic::CompareOp,
    exceptions::{PyNotImplementedError, PyValueError},
    prelude::*,
    PyNumberProtocol, PyObjectProtocol,
};
use std::fmt::Write;

use crate::ndarray::{shape::Shape, NdArray};

// TODO: once this is stable use a macro to generate for a variaty of types...
//
#[pyclass]
#[derive(Debug)]
pub struct NdArrayD {
    inner: NdArray<f64>,
}

#[pyproto]
impl PyObjectProtocol for NdArrayD {
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
    // TODO: return bool array
    fn __richcmp__(&'p self, other: PyRef<'p, Self>, op: CompareOp) -> PyResult<NdArrayD> {
        // TODO: check shape and raise error on uncomparable shapes
        // TODO: support different shapes e.g. compare each element to a scalar
        let op: fn(f64, f64) -> bool = match op {
            CompareOp::Lt => |a, b| a < b,
            CompareOp::Le => |a, b| a <= b,
            CompareOp::Eq => |a, b| (a - b).abs() < std::f64::EPSILON,
            CompareOp::Ne => |a, b| (a - b).abs() >= std::f64::EPSILON,
            CompareOp::Gt => |a, b| a > b,
            CompareOp::Ge => |a, b| a >= b,
        };
        let values: Vec<_> = self
            .inner
            .as_slice()
            .iter()
            .zip(other.inner.as_slice().iter())
            .map(|(a, b)| if op(*a, *b) { 1.0 } else { 0.0 })
            .collect();
        let mut res = NdArray::<f64>::new_vector(values.into());
        res.reshape(self.inner.shape.clone())
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))?;
        Ok(Self { inner: res })
    }
}

#[pyproto]
impl PyNumberProtocol for NdArrayD {
    fn __matmul__(lhs: PyRef<'p, Self>, rhs: PyRef<'p, Self>) -> PyResult<Self> {
        lhs.matmul(&*rhs)
    }
}

#[pymethods]
impl NdArrayD {
    #[new]
    pub fn new(shape: Vec<u32>, values: Option<Vec<f64>>) -> PyResult<Self> {
        let inner = match values {
            Some(v) => NdArray::new_with_values(shape.into_boxed_slice(), v.into_boxed_slice())
                .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))?,
            None => NdArray::new(shape.into_boxed_slice()),
        };
        Ok(Self { inner })
    }

    pub fn shape(&self) -> Vec<u32> {
        match self.inner.shape() {
            Shape::Scalar => vec![],
            Shape::Vector(n) => vec![*n],
            Shape::Matrix(n, m) => vec![*n, *m],
            Shape::Nd(s) => s.clone().into_vec(),
        }
    }

    pub fn reshape(&mut self, new_shape: Vec<u32>) -> PyResult<()> {
        self.inner
            .reshape(Shape::from(new_shape))
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))?;
        Ok(())
    }

    pub fn get(&self, index: Vec<u32>) -> Option<f64> {
        self.inner.get(&index).cloned()
    }

    pub fn set(&mut self, index: Vec<u32>, value: f64) {
        if let Some(x) = self.inner.get_mut(&index) {
            *x = value;
        }
    }

    /// Return if all values are truthy
    pub fn all(&self) -> bool {
        self.inner.values.iter().all(|x| *x != 0.0)
    }

    /// Return if any value is truthy
    pub fn any(&self) -> bool {
        self.inner.values.iter().any(|x| *x != 0.0)
    }

    /// The values must have a length equal to the product of the dimensions!
    pub fn set_values(&mut self, values: Vec<f64>) -> PyResult<()> {
        self.inner
            .set_slice(values.into_boxed_slice())
            .map(|_| ())
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))
    }

    pub fn matmul(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .matmul(&other.inner)
            .map(|inner| Self { inner })
            .map_err(|err| PyValueError::new_err::<String>(format!("{}", err).into()))
    }

    // TODO __str__
    pub fn to_string(&self) -> String {
        let depth = match self.inner.shape() {
            Shape::Scalar => {
                return format!("Scalar: {:?}", self.inner.get(&[]));
            }
            Shape::Vector(_) => 1,
            Shape::Matrix(_, _) => 2,
            Shape::Nd(s) => s.len(),
        };
        let mut s = String::with_capacity(self.inner.len() * 4);
        for _ in 0..depth - 1 {
            s.push('[');
        }
        let mut it = self.inner.iter_cols();
        if let Some(col) = it.next() {
            write!(s, "{:?}", col).unwrap();
        }
        for col in it {
            s.push('\n');
            for _ in 0..depth - 1 {
                s.push(' ');
            }
            write!(s, "{:?}", col).unwrap();
        }
        for _ in 0..depth - 1 {
            s.push(']');
        }
        s
    }
}
