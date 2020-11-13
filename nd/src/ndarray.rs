pub mod shape;

mod scalar;
pub use scalar::*;

#[cfg(test)]
mod tests;

use std::{mem::MaybeUninit, ops::Add, ops::Mul};

use shape::NdArrayShape;

#[derive(thiserror::Error, Debug)]
pub enum NdArrayError {
    #[error("DimensionMismatch error, expected: {expected}, actual: {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

#[derive(Debug)]
pub struct NdArray<T> {
    shape: NdArrayShape,
    values: Box<[T]>,
}

unsafe impl<T> Send for NdArray<T> {}

impl<'a, T> NdArray<T>
where
    T: Add<Output = T> + Mul<Output = T> + Default + 'a,
    &'a T: Add<Output = T> + 'a,
    &'a T: Mul<Output = T> + 'a,
{
    /// Ordinary inner product of vectors for 1-D arrays (without complex conjugation),
    /// in higher dimensions a sum product over the last axes.
    ///
    /// Returns None if the shapes are invalid
    pub fn inner(&'a self, other: &'a Self) -> Option<T> {
        match (&self.shape, &other.shape) {
            // scalar * scalar
            (NdArrayShape::Scalar, NdArrayShape::Scalar) => {
                return self
                    .values
                    .get(0)
                    .and_then(|a| other.values.get(0).map(|b| a * b))
            }
            // multiply the internal array with the scalar and sum it
            (NdArrayShape::Scalar, NdArrayShape::Matrix(_, _))
            | (NdArrayShape::Scalar, NdArrayShape::Nd(_))
            | (NdArrayShape::Scalar, NdArrayShape::Vector(_)) => {
                return self.values.get(0).map(|a| {
                    other
                        .values
                        .iter()
                        .fold(T::default(), |res, b| res + (a * b))
                })
            }
            // multiply the internal array with the scalar and sum it
            (NdArrayShape::Vector(_), NdArrayShape::Scalar)
            | (NdArrayShape::Matrix(_, _), NdArrayShape::Scalar)
            | (NdArrayShape::Nd(_), NdArrayShape::Scalar) => {
                return other.values.get(0).map(|a| {
                    self.values
                        .iter()
                        .fold(T::default(), |res, b| res + (a * b))
                })
            }

            // ordinary inner-product
            (NdArrayShape::Vector(_), NdArrayShape::Vector(_)) => {
                let val = self
                    .values
                    .iter()
                    .zip(other.values.iter())
                    .map(|(a, b)| a * b)
                    .fold(Default::default(), |a: T, b| a + b);
                return Some(val);
            }

            // sum over the columns, vector products
            (NdArrayShape::Matrix(c, _), NdArrayShape::Vector(n))
            | (NdArrayShape::Vector(n), NdArrayShape::Matrix(c, _)) => {
                if c != n {
                    return None;
                }
            }

            (NdArrayShape::Nd(shape), NdArrayShape::Vector(n))
            | (NdArrayShape::Vector(n), NdArrayShape::Nd(shape)) => {
                if shape.last()? != n {
                    return None;
                }
            }

            // Frobenius inner product
            //
            (NdArrayShape::Matrix(c1, r1), NdArrayShape::Matrix(c2, r2)) => {
                if c1 != c2 || r1 != r2 {
                    return None;
                }
            }

            // Frobenius inner product for nd arrays
            //
            (NdArrayShape::Matrix(c, r), NdArrayShape::Nd(s))
            | (NdArrayShape::Nd(s), NdArrayShape::Matrix(c, r)) => {
                if s.last()? != r {
                    return None;
                }
                let num_cols: u32 = s[..s.len() - 1].iter().product();
                if num_cols != *c {
                    return None;
                }
            }
            (NdArrayShape::Nd(sa), NdArrayShape::Nd(sb)) => {
                // column size mismatch
                if sa.last()? != sb.last()? {
                    return None;
                }
                // number of columns mismatch
                let num_cols_a: u32 = sa[..sa.len() - 1].iter().product();
                let num_cols_b: u32 = sb[..sb.len() - 1].iter().product();
                if num_cols_a != num_cols_b {
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

        let shape = match shape.len() {
            0 | 1 if shape[0] == 0 => NdArrayShape::Scalar,
            1 => NdArrayShape::Vector(shape[0]),
            2 => NdArrayShape::Matrix(shape[0], shape[1]),
            _ => NdArrayShape::Nd(shape),
        };

        Self {
            shape,
            values: values.into_boxed_slice(),
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

        let shape = match shape.len() {
            0 => NdArrayShape::Scalar,
            1 => NdArrayShape::Vector(shape[0]),
            2 => NdArrayShape::Matrix(shape[0], shape[1]),
            _ => NdArrayShape::Nd(shape),
        };
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

    pub fn shape(&self) -> &NdArrayShape {
        &self.shape
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
            NdArrayShape::Scalar => self.values.get(0),
            NdArrayShape::Vector(_) => self.values.get(*index.get(0)? as usize),
            NdArrayShape::Matrix(n, m) => {
                let i = get_index(1, &[*n, *m], index)?;
                self.values.get(i)
            }
            NdArrayShape::Nd(shape) => {
                let i = get_index(1, shape, index)?;
                self.values.get(i)
            }
        }
    }

    pub fn get_mut(&mut self, index: &[u32]) -> Option<&mut T> {
        match &self.shape {
            NdArrayShape::Scalar => self.values.get_mut(0),
            NdArrayShape::Vector(_) => self.values.get_mut(*index.get(0)? as usize),
            NdArrayShape::Matrix(n, m) => {
                let i = get_index(1, &[*n, *m], index)?;
                self.values.get_mut(i)
            }
            NdArrayShape::Nd(shape) => {
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
            NdArrayShape::Scalar | NdArrayShape::Vector(_) => Some(&self.values),
            NdArrayShape::Matrix(n, m) => {
                let m = *m as usize;
                let i = get_index(
                    m,
                    &[*n], // skip the last dim
                    index,
                )?;
                let i = i * m;
                Some(&self.as_slice()[i..i + m])
            }
            NdArrayShape::Nd(shape) => {
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
            NdArrayShape::Scalar | NdArrayShape::Vector(_) => Some(&mut self.values),
            NdArrayShape::Matrix(n, m) => {
                let m = *m as usize;
                let i = get_index(
                    m,
                    &[*n], // skip the last dim
                    index,
                )?;
                let i = i * m;
                Some(&mut self.as_mut_slice()[i..i + m])
            }
            NdArrayShape::Nd(shape) => {
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
