//! Basic arithmetic operations
//!
use std::ops::{Add, AddAssign};

use super::{column_iter::ColumnIterMut, shape::Shape, NdArray, NdArrayError};

impl<'a, T> NdArray<T>
where
    &'a T: Add<&'a T>,
    T: Add<T, Output = T> + AddAssign + Copy + 'a,
{
    pub fn add(&self, rhs: &Self) -> Result<Self, NdArrayError> {
        match (&self.shape, &rhs.shape) {
            (Shape::Scalar, Shape::Scalar) => Ok(NdArray::<T> {
                values: [self.values[0] + rhs.values[0]].into(),
                shape: Shape::Scalar,
            }),
            // add the scalar to all elements
            (Shape::Scalar, Shape::Vector(_))
            | (Shape::Scalar, Shape::Matrix(_, _))
            | (Shape::Scalar, Shape::Tensor(_)) => {
                let mut res = rhs.clone();
                let val = self.values[0];
                res.values.iter_mut().for_each(|x| *x += val);
                Ok(res)
            }
            (Shape::Vector(_), Shape::Scalar)
            | (Shape::Matrix(_, _), Shape::Scalar)
            | (Shape::Tensor(_), Shape::Scalar) => {
                let mut res = self.clone();
                let val = rhs.values[0];
                res.values.iter_mut().for_each(|x| *x += val);
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
                let values: Vec<_> = self
                    .values
                    .iter()
                    .zip(rhs.values.iter())
                    .map(|(a, b)| *a + *b)
                    .collect();
                let res = NdArray::<T> {
                    shape: self.shape.clone(),
                    values: values.into_boxed_slice(),
                };
                Ok(res)
            }

            // add vector to each column
            (Shape::Vector(l), Shape::Matrix(_, _)) | (Shape::Vector(l), Shape::Tensor(_)) => {
                let l = *l;
                if rhs.shape.last().expect("failed to get column shape") != l {
                    return Err(NdArrayError::DimensionMismatch {
                        expected: l as usize,
                        actual: rhs.shape.last().expect("failed to get column shape") as usize,
                    });
                }
                let mut res = rhs.clone();
                for col in res.iter_cols_mut() {
                    for (a, b) in col.iter_mut().zip(self.values.iter()) {
                        *a += *b;
                    }
                }
                Ok(res)
            }
            (Shape::Matrix(_, _), Shape::Vector(l)) | (Shape::Tensor(_), Shape::Vector(l)) => {
                let l = *l;
                if self.shape.last().expect("failed to get column shape") != l {
                    return Err(NdArrayError::DimensionMismatch {
                        expected: l as usize,
                        actual: self.shape.last().expect("failed to get column shape") as usize,
                    });
                }
                let mut res = self.clone();
                for col in res.iter_cols_mut() {
                    for (a, b) in col.iter_mut().zip(rhs.values.iter()) {
                        *a += *b;
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
                        expected: self.shape.clone(),
                        actual: shp.clone(),
                    });
                }

                let mut res = rhs.clone();
                for submat in ColumnIterMut::new(&mut res.values, k as usize * l as usize) {
                    submat
                        .iter_mut()
                        .zip(self.values.iter())
                        .for_each(|(a, b)| {
                            *a += *b;
                        })
                }
                Ok(res)
            }
            (shp @ Shape::Tensor(_), Shape::Matrix(n, m)) => {
                let [n, m] = [*n, *m];
                let [k, l] = shp.last_two().unwrap();
                if n != k || m != l {
                    return Err(NdArrayError::ShapeMismatch {
                        expected: self.shape.clone(),
                        actual: shp.clone(),
                    });
                }

                let mut res = self.clone();
                for submat in ColumnIterMut::new(&mut res.values, k as usize * l as usize) {
                    submat.iter_mut().zip(rhs.values.iter()).for_each(|(a, b)| {
                        *a += *b;
                    })
                }
                Ok(res)
            }
        }
    }
}
