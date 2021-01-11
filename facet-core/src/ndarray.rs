pub mod column_iter;
pub mod matrix;
pub mod shape;

mod arithmetic;
mod scalar;
use column_iter::{ColumnIter, ColumnIterMut};
pub use scalar::*;
use smallvec::SmallVec;

#[cfg(test)]
mod tests;

use std::{
    fmt::Debug, fmt::Write, iter::FromIterator, mem::MaybeUninit, ops::Add, ops::AddAssign,
    ops::Mul,
};

use shape::Shape;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(thiserror::Error, Debug)]
pub enum NdArrayError {
    #[error("DimensionMismatch error, expected: {expected}, actual: {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("DimensionMismatch error, expected: {expected:?}, actual: {actual:?}")]
    ShapeMismatch { expected: Shape, actual: Shape },
    #[error("Binary operation between the given shapes is not supported. Shape A: {shape_a:?} Shape B: {shape_b:?}")]
    BinaryOpNotSupported { shape_a: Shape, shape_b: Shape },
    #[error("Failed to convert value type into another. {0}")]
    ConversionError(String),
    #[error("Shape {0:?} is not supported for this operation")]
    UnsupportedShape(Shape),
    #[error("Invalid input given. {0}")]
    BadInput(String),
}

pub type Data<T> = SmallVec<[T; 16]>;
pub type Stride = SmallVec<[usize; 4]>;

/// Dense array of values
#[derive(Debug, Eq, PartialEq)]
pub struct NdArray<T> {
    shape: Shape,
    stride: Stride,
    values: Data<T>,
}

impl<T> Default for NdArray<T>
where
    T: Default,
{
    fn default() -> Self {
        let shape = Shape::Scalar(Default::default());
        let stride = shape::stride_vec(1, shape.as_slice());
        let mut values = Data::new();
        values.push(Default::default());
        Self {
            shape,
            stride,
            values,
        }
    }
}

impl<T> Clone for NdArray<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.clone(),
            stride: self.stride.clone(),
        }
    }
}

unsafe impl<T> Send for NdArray<T> {}

impl<'a, T> NdArray<T>
where
    T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a + Copy,
{
    /// Ordinary inner product of vectors for 1-D arrays (without complex conjugation),
    /// in higher dimensions a sum product over the last axes.
    ///
    /// Returns None if the shapes are invalid
    pub fn inner(&'a self, other: &'a Self) -> Option<T> {
        match (&self.shape, &other.shape) {
            // scalar * scalar
            (Shape::Scalar(_), Shape::Scalar(_)) => {
                return self
                    .values
                    .get(0)
                    .and_then(|a| other.values.get(0).map(|b| *a * *b))
            }
            // multiply the internal array with the scalar and sum it
            (Shape::Scalar(_), Shape::Matrix([_, _]))
            | (Shape::Scalar(_), Shape::Tensor(_))
            | (Shape::Scalar(_), Shape::Vector(_)) => {
                return self.values.get(0).map(|a| {
                    other
                        .values
                        .iter()
                        .fold(T::default(), |res, b| res + (*a * *b))
                })
            }
            // multiply the internal array with the scalar and sum it
            (Shape::Vector(_), Shape::Scalar(_))
            | (Shape::Matrix([_, _]), Shape::Scalar(_))
            | (Shape::Tensor(_), Shape::Scalar(_)) => {
                return other.values.get(0).map(|a| {
                    self.values
                        .iter()
                        .fold(T::default(), |res, b| res + (*a * *b))
                })
            }

            // ordinary inner-product
            (Shape::Vector(_), Shape::Vector(_)) => {
                let val = self
                    .values
                    .iter()
                    .zip(other.values.iter())
                    .map(|(a, b)| *a * *b)
                    .fold(Default::default(), |a: T, b| a + b);
                return Some(val);
            }

            // sum over the columns, vector products
            (Shape::Matrix([c, _]), Shape::Vector([n]))
            | (Shape::Vector([n]), Shape::Matrix([c, _])) => {
                if *c != *n {
                    return None;
                }
            }

            (Shape::Tensor(shape), Shape::Vector([n]))
            | (Shape::Vector([n]), Shape::Tensor(shape)) => {
                if *shape.last()? != *n {
                    return None;
                }
            }

            // Frobenius inner product
            //
            (Shape::Matrix([c1, r1]), Shape::Matrix([c2, r2])) => {
                if c1 != c2 || r1 != r2 {
                    return None;
                }
            }

            // Frobenius inner product for nd arrays
            //
            (Shape::Matrix([c, r]), Shape::Tensor(s))
            | (Shape::Tensor(s), Shape::Matrix([c, r])) => {
                if s.last()? != r {
                    return None;
                }
                let num_cols: u32 = s[..s.len() - 1].iter().product();
                if num_cols != *c {
                    return None;
                }
            }
            (Shape::Tensor(sa), Shape::Tensor(sb)) => {
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
                    .map(|(a, b)| *a * *b)
                    .fold(T::default(), |a, b| a + b)
            })
            // sum over the dot product result vector
            .fold(T::default(), |a, b| a + b);
        Some(res)
    }
}

impl<T> NdArray<T> {
    /// Created a new array with shape and uninitialized data.
    pub fn new(shape: impl Into<Shape>) -> Self
    where
        T: Copy,
    {
        let shape = shape.into();
        let len: usize = shape.span();
        let values = (0..len)
            .map(|_| unsafe { MaybeUninit::uninit().assume_init() })
            .collect();

        Self::new_with_values(shape, values).unwrap()
    }

    /// In the case of Tensors transpose the inner matrices
    /// e.g. Shape `[3, 4, 5]` becomes `[3, 5, 4]`
    pub fn transpose(self) -> Self
    where
        T: Send + Sync + Copy,
    {
        match &self.shape {
            Shape::Scalar(_) => self,
            Shape::Vector([n]) => Self::new_with_values([*n, 1], self.values).unwrap(),
            Shape::Matrix([n, m]) => {
                let mut values = self.values.clone();
                matrix::transpose_mat([*n as usize, *m as usize], &self.values, &mut values);
                Self::new_with_values(Shape::Matrix([*m, *n]), values).unwrap()
            }
            shape @ Shape::Tensor(_) => {
                // inner matrix tmp
                let shape_len = shape.as_slice().len();
                let [n, m] = shape.last_two().unwrap();
                let [m, n] = [m as usize, n as usize];
                let mut tmp = Vec::with_capacity(n * m);
                tmp.extend_from_slice(&self.values[..n * m]);

                // TODO
                // 2 million iq move: copy the first slice then use the first slice as the tmp
                //   array

                let mut values = Data::with_capacity(shape.span());
                for submat in ColumnIter::new(&self.values, n * m) {
                    matrix::transpose_mat([n, m], submat, &mut tmp);
                    values.extend_from_slice(&tmp);
                }

                let mut shape = shape.clone();
                shape[shape_len - 1] = n as u32;
                shape[shape_len - 2] = m as u32;

                Self::new_with_values(shape, values).unwrap()
            }
        }
    }

    pub fn new_scalar(value: T) -> Self
    where
        T: Clone,
    {
        let shape = Shape::Scalar(Default::default());
        let stride = shape::stride_vec(1, shape.as_slice());
        Self {
            values: [value][..].into(),
            stride,
            shape,
        }
    }
    pub fn new_with_values<S: Into<Shape>>(
        shape: S,
        values: Data<T>,
    ) -> Result<Self, NdArrayError> {
        let shape = shape.into();

        let len: usize = shape.span();
        if len != 0 && values.len() != len {
            return Err(NdArrayError::DimensionMismatch {
                expected: len,
                actual: values.len(),
            });
        }

        let res = Self {
            stride: shape::stride_vec(1, shape.as_slice()),
            shape,
            values,
        };
        Ok(res)
    }

    /// Construct a new 'vector' type (1D) array
    pub fn new_vector(values: impl Into<Data<T>>) -> Self {
        let values = values.into();
        Self::new_with_values(values.len() as u32, values).unwrap()
    }

    pub fn new_default<S: Into<Shape>>(shape: S) -> Self
    where
        T: Default,
    {
        let shape: Shape = shape.into();
        let len: usize = shape.span().max(1);
        let values = (0..len).map(|_| Default::default()).collect::<Data<T>>();
        Self::new_with_values(shape, values).unwrap()
    }

    /// Returns a Diagonal matrix where the values are the given default value and the rest of the
    /// items are Default'ed
    pub fn diagonal(columns: u32, default: T) -> Self
    where
        T: Default + Clone,
    {
        let mut res = Self::new_default([columns, columns]);
        let columns = columns as usize;
        for i in 0..columns {
            res.values[i * columns + i] = default.clone()
        }
        res
    }

    /// If invalid returns false and leaves this instance unchanged
    pub fn set_slice(&mut self, values: Data<T>) -> Result<&mut Self, NdArrayError> {
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

    pub fn stride(&self) -> &Stride {
        &self.stride
    }

    /// Smaller shape span will result in the last items being 'cut'
    pub fn reshape(&mut self, new_shape: impl Into<Shape>) -> &mut Self
    where
        T: Default,
    {
        let new_shape = new_shape.into();
        let new_len = new_shape.span();
        self.values.resize_with(new_len, Default::default);
        self.shape = new_shape;
        self
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
            Shape::Scalar(_) => self.values.get(0),
            Shape::Vector(_) => self.values.get(*index.get(0)? as usize),
            Shape::Matrix([n, m]) => {
                let i = get_index(&[*n, *m], &[*m as usize, 1], index)?;
                self.values.get(i)
            }
            Shape::Tensor(shape) => {
                let i = get_index(shape, &self.stride, index)?;
                self.values.get(i)
            }
        }
    }

    /// Returns `None` on invalid index
    pub fn get_mut(&mut self, index: &[u32]) -> Option<&mut T> {
        match &self.shape {
            Shape::Scalar(_) => self.values.get_mut(0),
            Shape::Vector(_) => self.values.get_mut(*index.get(0)? as usize),
            Shape::Matrix([n, m]) => {
                let i = get_index(&[*n, *m], &[*m as usize, 1], index)?;
                self.values.get_mut(i)
            }
            Shape::Tensor(shape) => {
                let i = get_index(shape, &self.stride, index)?;
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
    /// use facet_core::ndarray::NdArray;
    ///
    /// let mut arr = NdArray::<i32>::new(&[4, 2, 3][..]);
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
            Shape::Scalar(_) | Shape::Vector(_) => Some(&self.values),
            Shape::Matrix([n, m]) => {
                let i = get_index(
                    &[*n], // skip the last dim
                    self.stride.as_slice(),
                    index,
                )?;
                let m = *m as usize;
                Some(&self.as_slice()[i..i + m])
            }
            Shape::Tensor(shape) => {
                let i = get_index(
                    &shape[..shape.len() - 1], // skip the last dim
                    &self.stride[..shape.len() - 1],
                    index,
                )?;
                let w = self.shape.last().unwrap() as usize;
                Some(&self.as_slice()[i..i + w])
            }
        }
    }

    pub fn get_column_mut(&mut self, index: &[u32]) -> Option<&mut [T]> {
        match &self.shape {
            Shape::Scalar(_) | Shape::Vector(_) => Some(&mut self.values),
            Shape::Matrix([n, m]) => {
                let i = get_index(
                    &[*n], // skip the last dim
                    &[1],
                    index,
                )?;
                let m = *m as usize;
                Some(&mut self.as_mut_slice()[i..i + m])
            }
            Shape::Tensor(shape) => {
                let i = get_index(
                    &shape[..shape.len() - 1], // skip the last dim
                    &self.stride[..shape.len() - 1],
                    index,
                )?;
                let w = *shape.last().unwrap() as usize;
                Some(&mut self.as_mut_slice()[i..i + w])
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.values.iter()
    }

    pub fn iter_cols(&self) -> impl Iterator<Item = &[T]> {
        ColumnIter::new(&self.values, self.shape.last().unwrap_or(1) as usize)
    }

    pub fn iter_cols_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        ColumnIterMut::new(&mut self.values, self.shape.last().unwrap_or(1) as usize)
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_cols(&self) -> impl rayon::prelude::ParallelIterator<Item = &[T]> + '_
    where
        T: Sync,
    {
        let cols = self.shape.last().unwrap_or(1) as usize;
        self.values.as_slice().par_chunks(cols)
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_cols_mut(
        &mut self,
    ) -> impl rayon::prelude::ParallelIterator<Item = &mut [T]> + '_
    where
        T: Sync + Send,
    {
        let cols = self.shape.last().unwrap_or(1) as usize;
        self.values.as_mut_slice().par_chunks_mut(cols)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// fold the index
#[inline]
fn get_index(shape: &[u32], stride: &[usize], index: &[u32]) -> Option<usize> {
    if index.len() > shape.len() {
        return None;
    }
    let mut res = 0;
    for i in 0..index.len() {
        if index[i] >= shape[i] {
            return None;
        }
        let skip = index[i] as usize;
        res += skip * stride[i];
    }

    Some(res)
}

impl<T> From<T> for NdArray<T>
where
    T: Copy,
{
    fn from(val: T) -> Self {
        NdArray::new_with_values(0, Data::from_slice(&[val][..])).unwrap()
    }
}

impl<'a, T> FromIterator<T> for NdArray<T> {
    fn from_iter<It: IntoIterator<Item = T>>(iter: It) -> Self {
        let values: Data<T> = iter.into_iter().collect();
        Self::new_with_values(values.len() as u32, values).unwrap()
    }
}

impl<T> std::fmt::Display for NdArray<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO: fix Tensor printing, currently the inner `[]`'s aren't printed.
        // maybe do a recursive function?
        let depth = match self.shape() {
            Shape::Scalar(_) => {
                return write!(f, "{:?}", self.get(&[]).unwrap());
            }
            Shape::Vector(_) => 1,
            Shape::Matrix(_) => 2,
            Shape::Tensor(s) => s.len(),
        };
        for _ in 0..depth - 1 {
            f.write_char('[')?;
        }
        let mut it = self.iter_cols();
        if let Some(col) = it.next() {
            write!(f, "{:.5?}", col).unwrap();
        }
        for col in it {
            f.write_char('\n')?;
            for _ in 0..depth - 1 {
                f.write_char(' ')?;
            }
            write!(f, "{:.5?}", col).unwrap();
        }
        for _ in 0..depth - 1 {
            f.write_char(']')?;
        }
        Ok(())
    }
}
