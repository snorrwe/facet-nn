//! Matrix operation implementations
//!

use std::ops::{Add, AddAssign, Mul};

use super::{NdArray, NdArrayError, column_iter::ColumnIter, column_iter::ColumnMutIter, shape::Shape};

fn matmul_impl<'a, T>(
    [n, m]: [u32; 2],
    values0: &'a [T],
    [m1, p]: [u32; 2],
    values1: &'a [T],
    out: &mut [T],
) -> Result<(), NdArrayError>
where
    T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a,
    &'a T: Add<Output = T> + 'a + Mul<Output = T>,
{
    if m != m1 {
        return Err(NdArrayError::DimensionMismatch {
            expected: m as usize,
            actual: m1 as usize,
        });
    }
    debug_assert_eq!((n as usize * m as usize), values0.len());
    debug_assert_eq!((p as usize * m as usize), values1.len());
    debug_assert_eq!(out.len(), n as usize * p as usize);

    for i in 0..n {
        for j in 0..p {
            for k in 0..m {
                let val0 = &values0[(i * m + k) as usize];
                let val1 = &values1[(k * p + j) as usize];
                out[(i * p + j) as usize] += val0 * val1
            }
        }
    }

    Ok(())
}

impl<'a, T> NdArray<T>
where
    T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a,
    &'a T: Add<Output = T> + 'a + Mul<Output = T>,
{
    /// - Scalars not allowed.
    /// - Nd arrays are treated as a colection of matrices and are broadcast accordingly
    ///
    /// ## Nd arrays Example
    ///
    /// This example will take a single 2 by 3 matrix and multiply it with 2 3 by 2 matrices.
    /// The output is 2(!) 2 by 2 matrices.
    ///
    /// ```
    /// use nd::ndarray::NdArray;
    /// use nd::ndarray::shape::Shape;
    ///
    /// // 2 by 3 matrix
    /// let a = NdArray::new_with_values([2, 3].into(), [1, 2, -1, 2, 0, 1].into()).unwrap();
    ///
    /// // the same 3 by 2 matrix twice
    /// let b = NdArray::new_with_values(
    ///     [2, 3, 2].into(),
    ///     [3, 1, 0, -1, -2, 3, /*|*/ 3, 1, 0, -1, -2, 3].into(),
    /// )
    /// .unwrap();
    ///
    /// let c = a.matmul(&b).expect("matmul");
    ///
    /// // output 2 2 by 2 matrices
    /// assert_eq!(c.shape(), &Shape::Nd([2, 2, 2].into()));
    /// assert_eq!(c.as_slice(), &[5, -4, 4, 5, /*|*/ 5, -4, 4, 5]);
    /// ```
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
                let mut res = Self::new_default([1, *m].into());
                matmul_impl(
                    [1, *l],
                    self.as_slice(),
                    [*n, *m],
                    other.as_slice(),
                    res.as_mut_slice(),
                )?;
                res.reshape(Shape::Vector(*m))?;
                Ok(res)
            }
            (Shape::Matrix(n, m), Shape::Vector(l)) => {
                let mut res = Self::new_default([*n, 1].into());
                matmul_impl(
                    [*n, *m],
                    self.as_slice(),
                    [*l, 1],
                    other.as_slice(),
                    res.as_mut_slice(),
                )?;
                res.reshape(Shape::Vector(*m))?;
                Ok(res)
            }
            (Shape::Matrix(a, b), Shape::Matrix(c, d)) => {
                let mut res = Self::new_default([*a, *d].into());
                matmul_impl(
                    [*a, *b],
                    self.as_slice(),
                    [*c, *d],
                    other.as_slice(),
                    res.as_mut_slice(),
                )?;
                Ok(res)
            }

            // broadcast matrices
            (Shape::Vector(l), Shape::Nd(shp)) => {
                let [m, n] = [shp[shp.len() - 1], shp[shp.len() - 2]];

                let it = ColumnIter::new(&other.values, n as usize * m as usize);
                let mut out = Self::new_default(
                    [(other.len() / (n as usize * m as usize)) as u32, *l].into(),
                );
                for (mat, out) in it.zip(ColumnMutIter::new(&mut out.values, *l as usize)) {
                    matmul_impl([1, *l], self.as_slice(), [n, m], mat, out)?;
                }
                Ok(out)
            }
            (Shape::Nd(shp), Shape::Vector(l)) => {
                let [m, n] = [shp[shp.len() - 1], shp[shp.len() - 2]];

                let it = ColumnIter::new(&self.values, n as usize * m as usize);
                let mut out =
                    Self::new_default([(self.len() / (n as usize * m as usize)) as u32, *l].into());
                for (mat, out) in it.zip(ColumnMutIter::new(&mut out.values, *l as usize)) {
                    matmul_impl([n, m], mat, [*l, 1], other.as_slice(), out)?;
                }
                Ok(out)
            }
            (Shape::Matrix(a, b), Shape::Nd(shp)) => {
                let [a, b] = [*a, *b];
                let [d, c] = [shp[shp.len() - 1], shp[shp.len() - 2]];

                let it = ColumnIter::new(&other.values, c as usize * d as usize);
                let mut out = Self::new_default(
                    [(other.len() / (c as usize * d as usize)) as u32, a, d].into(),
                );
                for (mat, out) in
                    it.zip(ColumnMutIter::new(&mut out.values, a as usize * d as usize))
                {
                    matmul_impl([a, b], self.as_slice(), [c, d], mat, out)?;
                }
                Ok(out)
            }
            (Shape::Nd(shp), Shape::Matrix(c, d)) => {
                let [b, a] = [shp[shp.len() - 1], shp[shp.len() - 2]];
                let [c, d] = [*c, *d];

                let it = ColumnIter::new(&self.values, c as usize * d as usize);
                let mut out = Self::new_default(
                    [(self.len() / (c as usize * d as usize)) as u32, a, d].into(),
                );
                for (mat, out) in
                    it.zip(ColumnMutIter::new(&mut out.values, a as usize * d as usize))
                {
                    matmul_impl([a, b], self.as_slice(), [c, d], mat, out)?;
                }
                Ok(out)
            }
            (Shape::Nd(ab), Shape::Nd(cd)) => {
                let [b, a] = [ab[ab.len() - 1], ab[ab.len() - 2]];
                let [d, c] = [cd[cd.len() - 1], cd[cd.len() - 2]];

                // number of matrices
                let nmatrices = self.shape.span() / (b as usize * a as usize);
                let other_nmatrices = other.shape.span() / (c as usize * d as usize);
                if nmatrices != other_nmatrices {
                    // the two arrays have a different number of inner matrices
                    return Err(NdArrayError::DimensionMismatch {
                        expected: nmatrices,
                        actual: other_nmatrices,
                    });
                }

                let it_0 = ColumnIter::new(&self.values, a as usize * b as usize);
                let it_1 = ColumnIter::new(&other.values, c as usize * d as usize);

                let mut out = Self::new_default([nmatrices as u32, a, d].into());

                for (out, (lhs, rhs)) in
                    ColumnMutIter::new(&mut out.values, a as usize * d as usize).zip(it_0.zip(it_1))
                {
                    matmul_impl([a, b], lhs, [c, d], rhs, out)?;
                }

                Ok(out)
            }
        }
    }
}
