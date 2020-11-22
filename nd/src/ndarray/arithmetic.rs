//! Basic arithmetic operations
//!
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use super::{column_iter::ColumnIterMut, shape::Shape, NdArray, NdArrayError};

macro_rules! arithimpl {
    ($opeq: tt, $op: tt, $lhs: ident, $rhs: ident) =>{
        match (&$lhs.shape, &$rhs.shape) {
            (Shape::Scalar, Shape::Scalar) => Ok(NdArray::<T> {
                values: [$lhs.values[0] $op $rhs.values[0]].into(),
                shape: Shape::Scalar,
            }),
            // add the scalar to all elements
            (Shape::Scalar, Shape::Vector(_))
            | (Shape::Scalar, Shape::Matrix(_, _))
            | (Shape::Scalar, Shape::Tensor(_)) => {
                let mut res = $rhs.clone();
                let val = $lhs.values[0];
                res.values.iter_mut().for_each(|x| *x $opeq val);
                Ok(res)
            }
            (Shape::Vector(_), Shape::Scalar)
            | (Shape::Matrix(_, _), Shape::Scalar)
            | (Shape::Tensor(_), Shape::Scalar) => {
                let mut res = $lhs.clone();
                let val = $rhs.values[0];
                res.values.iter_mut().for_each(|x| *x $opeq val);
                Ok(res)
            }

            // Element-wise
            (a @ Shape::Matrix(_, _), b @ Shape::Matrix(_, _))
            | (a @ Shape::Tensor(_), b @ Shape::Tensor(_))
            | (a @ Shape::Vector(_), b @ Shape::Vector(_)) => {
                if a != b {
                    return Err(NdArrayError::ShapeMismatch {
                        expected: a.clone(),
                        actual: b.clone(),
                    });
                }
                let values: Vec<_> = $lhs
                    .values
                    .iter()
                    .zip($rhs.values.iter())
                    .map(|(a, b)| *a $op *b)
                    .collect();
                let res = NdArray::<T> {
                    shape: $lhs.shape.clone(),
                    values: values.into_boxed_slice(),
                };
                Ok(res)
            }

            // add vector to each column
            (Shape::Vector(l), Shape::Matrix(_, _)) | (Shape::Vector(l), Shape::Tensor(_)) => {
                let l = *l;
                if $rhs.shape.last().expect("failed to get column shape") as u64 != l {
                    return Err(NdArrayError::DimensionMismatch {
                        expected: l as usize,
                        actual: $rhs.shape.last().expect("failed to get column shape") as usize,
                    });
                }
                let mut res = $rhs.clone();
                for col in res.iter_cols_mut() {
                    for (a, b) in col.iter_mut().zip($lhs.values.iter()) {
                        *a $opeq *b;
                    }
                }
                Ok(res)
            }
            (Shape::Matrix(_, _), Shape::Vector(l)) | (Shape::Tensor(_), Shape::Vector(l)) => {
                let l = *l;
                if $lhs.shape.last().expect("failed to get column shape") as u64 != l {
                    return Err(NdArrayError::DimensionMismatch {
                        expected: l as usize,
                        actual: $lhs.shape.last().expect("failed to get column shape") as usize,
                    });
                }
                let mut res = $lhs.clone();
                for col in res.iter_cols_mut() {
                    for (a, b) in col.iter_mut().zip($rhs.values.iter()) {
                        *a $opeq *b;
                    }
                }
                Ok(res)
            }

            // add a matrix to each matrix in a tensor
            (Shape::Matrix(n, m), shp @ Shape::Tensor(_)) => {
                let [n, m] = [*n, *m];
                let [k, l] = shp.last_two().unwrap();
                if n != k || m != l {
                    return Err(NdArrayError::ShapeMismatch {
                        expected: $lhs.shape.clone(),
                        actual: shp.clone(),
                    });
                }

                let mut res = $rhs.clone();
                for submat in ColumnIterMut::new(&mut res.values, k as usize * l as usize) {
                    submat
                        .iter_mut()
                        .zip($lhs.values.iter())
                        .for_each(|(a, b)| {
                            *a $opeq *b;
                        })
                }
                Ok(res)
            }
            (shp @ Shape::Tensor(_), Shape::Matrix(n, m)) => {
                let [n, m] = [*n, *m];
                let [k, l] = shp.last_two().unwrap();
                if n != k || m != l {
                    return Err(NdArrayError::ShapeMismatch {
                        expected: $lhs.shape.clone(),
                        actual: shp.clone(),
                    });
                }

                let mut res = $lhs.clone();
                for submat in ColumnIterMut::new(&mut res.values, k as usize * l as usize) {
                    submat.iter_mut().zip($rhs.values.iter()).for_each(|(a, b)| {
                        *a $opeq *b;
                    })
                }
                Ok(res)
            }
        }
    }
}

impl<'a, T> NdArray<T>
where
    T: Add<T, Output = T> + AddAssign + Copy + 'a,
{
    pub fn add(&self, rhs: &Self) -> Result<Self, NdArrayError> {
        arithimpl!(+=, +, self, rhs)
    }
}

impl<'a, T> NdArray<T>
where
    T: Sub<T, Output = T> + SubAssign + Copy + 'a,
{
    pub fn sub(&self, rhs: &Self) -> Result<Self, NdArrayError> {
        arithimpl!(-=, -, self, rhs)
    }
}

impl<'a, T> NdArray<T>
where
    T: Mul<T, Output = T> + MulAssign + Copy + 'a,
{
    pub fn mul(&self, rhs: &Self) -> Result<Self, NdArrayError> {
        arithimpl!(*=, *, self, rhs)
    }
}

impl<'a, T> NdArray<T>
where
    T: Div<T, Output = T> + DivAssign + Copy + 'a,
{
    pub fn div(&self, rhs: &Self) -> Result<Self, NdArrayError> {
        arithimpl!(/=, /, self, rhs)
    }
}

impl<'a, T> NdArray<T> {
    /// Maps the current array to another array with the same shape
    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> NdArray<U> {
        let res: Vec<_> = self.values.iter().map(f).collect();
        NdArray {
            shape: self.shape.clone(),
            values: res.into_boxed_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map() {
        let a = NdArray::<i32>::new_with_values(
            [2, 4, 2].into(),
            [0, 69, 0, 69, 0, 69, 0, 0, 0, 0, 69, 0, 69, 0, 69, 0].into(),
        )
        .unwrap();

        let b: NdArray<bool> = a.map(|x| *x > 0);

        assert_eq!(a.shape, b.shape);
        assert_eq!(
            b.values.as_ref(),
            &[
                false, true, false, true, false, true, false, false, false, false, true, false,
                true, false, true, false
            ]
        );
    }
}
