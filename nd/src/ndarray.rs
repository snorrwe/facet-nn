pub mod shape;

mod matrix;
mod scalar;
use matrix::matmul;
pub use scalar::*;

#[cfg(test)]
mod tests;

use std::{mem::MaybeUninit, ops::Add, ops::AddAssign, ops::Mul};

use shape::Shape;

#[derive(thiserror::Error, Debug)]
pub enum NdArrayError {
    #[error("DimensionMismatch error, expected: {expected}, actual: {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Binary operation between the given shapes is not supported. Shape A: {shape_a:?} Shape B: {shape_b:?}")]
    BinaryOpNotSupported { shape_a: Shape, shape_b: Shape },
}

#[derive(Debug, Eq, PartialEq)]
pub struct NdArray<T> {
    pub(crate) shape: Shape,
    pub(crate) values: Box<[T]>,
}

unsafe impl<T> Send for NdArray<T> {}

impl<'a, T> NdArray<T>
where
    T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a,
    &'a T: Add<Output = T> + 'a + Mul<Output = T>,
{
    /// - Scalars not allowed.
    /// - Nd arrays are treated as a colection of matrices and are broadcast accordingly
    pub fn matmul(&'a self, other: &'a Self) -> Result<Self, NdArrayError> {
        match (&self.shape, &other.shape) {
            shapes @ (Shape::Scalar, Shape::Scalar)
            | shapes @ (Shape::Scalar, Shape::Vector(_))
            | shapes @ (Shape::Scalar, Shape::Matrix(_, _))
            | shapes @ (Shape::Scalar, Shape::Nd(_))
            | shapes @ (Shape::Vector(_), Shape::Scalar)
            | shapes @ (Shape::Matrix(_, _), Shape::Scalar)
            | shapes @ (Shape::Nd(_), Shape::Scalar) => Err(NdArrayError::BinaryOpNotSupported {
                shape_a: shapes.0.clone(),
                shape_b: shapes.1.clone(),
            }),

            (Shape::Vector(a), Shape::Vector(b)) => {
                let res = self.inner(other).ok_or(NdArrayError::DimensionMismatch {
                    expected: *a as usize,
                    actual: *b as usize,
                })?;
                Self::new_with_values([].into(), [res].into())
            }

            (Shape::Vector(l), Shape::Matrix(n, m)) => {
                let mut res = matmul([1, *l], self.as_slice(), [*n, *m], other.as_slice())?;
                res.reshape(Shape::Vector(*m))?;
                Ok(res)
            }
            (Shape::Matrix(n, m), Shape::Vector(l)) => {
                let mut res = matmul([*n, *m], self.as_slice(), [*l, 1], other.as_slice())?;
                res.reshape(Shape::Vector(*m))?;
                Ok(res)
            }
            (Shape::Matrix(a, b), Shape::Matrix(c, d)) => {
                matmul([*a, *b], self.as_slice(), [*c, *d], other.as_slice())
            }

            // broadcast matrices
            (Shape::Vector(_), Shape::Nd(_)) => todo!(),
            (Shape::Matrix(_, _), Shape::Nd(_)) => todo!(),
            (Shape::Nd(_), Shape::Vector(_)) => todo!(),
            (Shape::Nd(_), Shape::Matrix(_, _)) => todo!(),
            (Shape::Nd(_), Shape::Nd(_)) => todo!(),
        }
    }

    /// - If both are 1D arrays, `dot` equals `inner`
    /// - If both are 2D arrays, it's equal to `matmul`
    pub fn dot(&'a self, _other: &'a Self) -> Option<Self> {
        todo!()
    }

    /// Ordinary inner product of vectors for 1-D arrays (without complex conjugation),
    /// in higher dimensions a sum product over the last axes.
    ///
    /// Returns None if the shapes are invalid
    pub fn inner(&'a self, other: &'a Self) -> Option<T> {
        match (&self.shape, &other.shape) {
            // scalar * scalar
            (Shape::Scalar, Shape::Scalar) => {
                return self
                    .values
                    .get(0)
                    .and_then(|a| other.values.get(0).map(|b| a * b))
            }
            // multiply the internal array with the scalar and sum it
            (Shape::Scalar, Shape::Matrix(_, _))
            | (Shape::Scalar, Shape::Nd(_))
            | (Shape::Scalar, Shape::Vector(_)) => {
                return self.values.get(0).map(|a| {
                    other
                        .values
                        .iter()
                        .fold(T::default(), |res, b| res + (a * b))
                })
            }
            // multiply the internal array with the scalar and sum it
            (Shape::Vector(_), Shape::Scalar)
            | (Shape::Matrix(_, _), Shape::Scalar)
            | (Shape::Nd(_), Shape::Scalar) => {
                return other.values.get(0).map(|a| {
                    self.values
                        .iter()
                        .fold(T::default(), |res, b| res + (a * b))
                })
            }

            // ordinary inner-product
            (Shape::Vector(_), Shape::Vector(_)) => {
                let val = self
                    .values
                    .iter()
                    .zip(other.values.iter())
                    .map(|(a, b)| a * b)
                    .fold(Default::default(), |a: T, b| a + b);
                return Some(val);
            }

            // sum over the columns, vector products
            (Shape::Matrix(c, _), Shape::Vector(n)) | (Shape::Vector(n), Shape::Matrix(c, _)) => {
                if c != n {
                    return None;
                }
            }

            (Shape::Nd(shape), Shape::Vector(n)) | (Shape::Vector(n), Shape::Nd(shape)) => {
                if shape.last()? != n {
                    return None;
                }
            }

            // Frobenius inner product
            //
            (Shape::Matrix(c1, r1), Shape::Matrix(c2, r2)) => {
                if c1 != c2 || r1 != r2 {
                    return None;
                }
            }

            // Frobenius inner product for nd arrays
            //
            (Shape::Matrix(c, r), Shape::Nd(s)) | (Shape::Nd(s), Shape::Matrix(c, r)) => {
                if s.last()? != r {
                    return None;
                }
                let num_cols: u32 = s[..s.len() - 1].iter().product();
                if num_cols != *c {
                    return None;
                }
            }
            (Shape::Nd(sa), Shape::Nd(sb)) => {
                // column size mismatch
                if sa.last()? != sb.last()? {
                    return None;
                }
                let num_cols_a: u32 = sa[..sa.len() - 1].iter().product();
                let num_cols_b: u32 = sb[..sb.len() - 1].iter().product();
                if num_cols_a != num_cols_b {
                    // number of columns mismatch
                    return None;
                }
            }
        }

        let res = other
            .iter_cols()
            .map(|col| {
                // dot product result
                self.values
                    .iter()
                    .zip(col.iter())
                    .map(|(a, b)| a * b)
                    .fold(T::default(), |a, b| a + b)
            })
            // sum over the dot product result vector
            .fold(T::default(), |a, b| a + b);
        Some(res)
    }
}

impl<T> NdArray<T>
where
    T: Copy,
{
    pub fn new(shape: Box<[u32]>) -> Self {
        let len: usize = shape.iter().map(|x| *x as usize).product();
        let values = (0..len)
            .map(|_| unsafe { MaybeUninit::uninit().assume_init() })
            .collect::<Vec<_>>();

        let shape = Shape::from(shape);

        Self {
            shape,
            values: values.into_boxed_slice(),
        }
    }
}

impl<T> NdArray<T> {
    pub fn new_with_values(shape: Box<[u32]>, values: Box<[T]>) -> Result<Self, NdArrayError> {
        let len: usize = shape.iter().map(|x| *x as usize).product();

        if len != 0 && values.len() != len {
            return Err(NdArrayError::DimensionMismatch {
                expected: len,
                actual: values.len(),
            });
        }

        let shape = Shape::from(shape);
        Ok(Self { shape, values })
    }

    /// Construct a new 'vector' type (1D) array
    pub fn new_vector(values: Box<[T]>) -> Self {
        Self {
            shape: Shape::Vector(values.len() as u32),
            values,
        }
    }
}

impl<T> NdArray<T>
where
    T: Default,
{
    pub fn new_default(shape: Box<[u32]>) -> Self {
        let len: usize = shape.iter().map(|x| *x as usize).product();
        let values = (0..len).map(|_| Default::default()).collect::<Vec<_>>();

        let shape = Shape::from(shape);
        Self {
            shape,
            values: values.into_boxed_slice(),
        }
    }
}

impl<T> NdArray<T> {
    /// If invalid returns false and leaves this instance unchanged
    pub fn set_slice(&mut self, values: Box<[T]>) -> Result<&mut Self, NdArrayError> {
        if values.len() != self.values.len() {
            return Err(NdArrayError::DimensionMismatch {
                expected: self.values.len(),
                actual: values.len(),
            });
        }

        self.values = values;

        Ok(self)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn reshape(&mut self, new_shape: Shape) -> Result<&mut Self, NdArrayError> {
        let len = self.len();
        let new_len = new_shape.span();
        if len != new_len {
            return Err(NdArrayError::DimensionMismatch {
                expected: len,
                actual: new_len,
            });
        }
        self.shape = new_shape;
        Ok(self)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Returns `None` on invalid index
    pub fn get(&self, index: &[u32]) -> Option<&T> {
        match &self.shape {
            Shape::Scalar => self.values.get(0),
            Shape::Vector(_) => self.values.get(*index.get(0)? as usize),
            Shape::Matrix(n, m) => {
                let i = get_index(1, &[*n, *m], index)?;
                self.values.get(i)
            }
            Shape::Nd(shape) => {
                let i = get_index(1, shape, index)?;
                self.values.get(i)
            }
        }
    }

    pub fn get_mut(&mut self, index: &[u32]) -> Option<&mut T> {
        match &self.shape {
            Shape::Scalar => self.values.get_mut(0),
            Shape::Vector(_) => self.values.get_mut(*index.get(0)? as usize),
            Shape::Matrix(n, m) => {
                let i = get_index(1, &[*n, *m], index)?;
                self.values.get_mut(i)
            }
            Shape::Nd(shape) => {
                let i = get_index(1, shape, index)?;
                self.values.get_mut(i)
            }
        }
    }

    /// Index must be N-1 long. Will return a vector
    ///
    /// E.g for a 1D array the column is the array itself
    /// for a 2D array, with shape (2, 5) the column is a 5D vector
    ///
    /// Returns `None` on invalid index.
    ///
    /// ```
    /// use nd::ndarray::NdArray;
    ///
    /// let mut arr = NdArray::<i32>::new([4, 2, 3].into());
    /// arr.as_mut_slice()
    ///     .iter_mut()
    ///     .enumerate()
    ///     .for_each(|(i, x)| *x = i as i32);
    ///
    /// let col = arr.get_column(&[1, 1]).unwrap();
    ///
    /// assert_eq!(col, &[9, 10, 11]);
    /// ```
    pub fn get_column(&self, index: &[u32]) -> Option<&[T]> {
        match &self.shape {
            Shape::Scalar | Shape::Vector(_) => Some(&self.values),
            Shape::Matrix(n, m) => {
                let m = *m as usize;
                let i = get_index(
                    m,
                    &[*n], // skip the last dim
                    index,
                )?;
                let i = i * m;
                Some(&self.as_slice()[i..i + m])
            }
            Shape::Nd(shape) => {
                let w = *shape.last().unwrap() as usize;
                let i = get_index(
                    w,
                    &shape[..shape.len() - 1], // skip the last dim
                    index,
                )?;
                let i = i * w;
                Some(&self.as_slice()[i..i + w])
            }
        }
    }

    pub fn get_column_mut(&mut self, index: &[u32]) -> Option<&mut [T]> {
        match &self.shape {
            Shape::Scalar | Shape::Vector(_) => Some(&mut self.values),
            Shape::Matrix(n, m) => {
                let m = *m as usize;
                let i = get_index(
                    m,
                    &[*n], // skip the last dim
                    index,
                )?;
                let i = i * m;
                Some(&mut self.as_mut_slice()[i..i + m])
            }
            Shape::Nd(shape) => {
                let w = *shape.last().unwrap() as usize;
                let i = get_index(
                    w,
                    &shape[..shape.len() - 1], // skip the last dim
                    index,
                )?;
                let i = i * w;
                Some(&mut self.as_mut_slice()[i..i + w])
            }
        }
    }

    pub fn iter_cols(&self) -> impl Iterator<Item = &[T]> {
        ColumnIterator::new(&self.values, self.shape.last().unwrap_or(0) as usize)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct ColumnIterator<'a, T> {
    arr: &'a [T],
    column_size: usize,
    current: usize,
}

impl<'a, T> ColumnIterator<'a, T> {
    pub fn new(arr: &'a [T], column_size: usize) -> Self {
        Self {
            arr,
            column_size,
            current: 0,
        }
    }
}

impl<'a, T> Iterator for ColumnIterator<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        let tail = self.current + self.column_size;
        if tail <= self.arr.len() {
            let res = &self.arr[self.current..tail];
            self.current = tail;
            Some(res)
        } else {
            None
        }
    }
}

/// fold the index
#[inline]
fn get_index(width: usize, shape: &[u32], index: &[u32]) -> Option<usize> {
    // they should be equal to avoid confusion, but this is enough to work
    let mut it = shape.iter().rev();
    let mut a = *it.next()? as usize;

    let mut res = *index.last()? as usize;
    let iit = index.iter().rev();

    for (dim, ind) in it.zip(iit).skip(1) {
        a *= *dim as usize;
        res += a * *ind as usize;
    }
    Some(res * width)
}
