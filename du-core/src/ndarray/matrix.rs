//! Matrix operation implementations
//!

use std::{
    convert::TryInto,
    ops::{Add, AddAssign, Mul},
};

use super::{
    column_iter::ColumnIter, column_iter::ColumnIterMut, shape::Shape, NdArray, NdArrayError,
};

fn matmul_impl<'a, T>(
    [n, m]: [u32; 2],
    values0: &'a [T],
    [m1, p]: [u32; 2],
    values1: &'a [T],
    out: &mut [T],
) -> Result<(), NdArrayError>
where
    T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a + Copy,
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
                out[(i * p + j) as usize] += *val0 * *val1
            }
        }
    }

    Ok(())
}

pub fn transpose_mat<T: Clone>([n, m]: [usize; 2], inp: &[T], out: &mut [T]) {
    for (i, col) in ColumnIter::new(inp, m).enumerate() {
        for (j, v) in col.iter().cloned().enumerate() {
            out[j * n + i] = v;
        }
    }
}

impl<'a, T> NdArray<T>
where
    T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a + Copy,
{
    /// - Scalars not allowed.
    /// - Tensor arrays are treated as a colection of matrices and are broadcast accordingly
    ///
    /// ## Tensor arrays Example
    ///
    /// This example will take a single 2 by 3 matrix and multiply it with 2 3 by 2 matrices.
    /// The output is 2(!) 2 by 2 matrices.
    ///
    /// ```
    /// use du_core::ndarray::NdArray;
    /// use du_core::ndarray::shape::Shape;
    ///
    /// // 2 by 3 matrix
    /// let a = NdArray::new_with_values([2, 3], [1, 2, -1, 2, 0, 1]).unwrap();
    ///
    /// // the same 3 by 2 matrix twice
    /// let b = NdArray::new_with_values(
    ///     &[2, 3, 2][..],
    ///     [3, 1, 0, -1, -2, 3, /*|*/ 3, 1, 0, -1, -2, 3],
    /// )
    /// .unwrap();
    ///
    /// let c = a.matmul(&b).expect("matmul");
    ///
    /// // output 2 2 by 2 matrices
    /// assert_eq!(c.shape(), &Shape::Tensor([2, 2, 2].into()));
    /// assert_eq!(c.as_slice(), &[5, -4, 4, 5, /*|*/ 5, -4, 4, 5]);
    /// ```
    pub fn matmul(&'a self, other: &'a Self) -> Result<Self, NdArrayError> {
        match (&self.shape, &other.shape) {
            shapes @ (Shape::Scalar, Shape::Scalar)
            | shapes @ (Shape::Scalar, Shape::Vector(_))
            | shapes @ (Shape::Scalar, Shape::Matrix(_, _))
            | shapes @ (Shape::Scalar, Shape::Tensor(_))
            | shapes @ (Shape::Vector(_), Shape::Scalar)
            | shapes @ (Shape::Matrix(_, _), Shape::Scalar)
            | shapes @ (Shape::Tensor(_), Shape::Scalar) => {
                Err(NdArrayError::BinaryOpNotSupported {
                    shape_a: shapes.0.clone(),
                    shape_b: shapes.1.clone(),
                })
            }

            (Shape::Vector(a), Shape::Vector(b)) => {
                let res = self.inner(other).ok_or(NdArrayError::DimensionMismatch {
                    expected: *a as usize,
                    actual: *b as usize,
                })?;
                Self::new_with_values(&[][..], [res])
            }

            (Shape::Vector(l), Shape::Matrix(n, m)) => {
                let mut res = Self::new_default(Shape::Matrix(1, *m));
                matmul_impl(
                    [1, (*l).try_into().unwrap()],
                    self.as_slice(),
                    [*n, *m],
                    other.as_slice(),
                    res.as_mut_slice(),
                )?;
                res.reshape(Shape::Vector(*m as u64))?;
                Ok(res)
            }
            (Shape::Matrix(n, m), Shape::Vector(l)) => {
                let mut res = Self::new_default(Shape::Matrix(*n, 1));
                matmul_impl(
                    [*n, *m],
                    self.as_slice(),
                    [(*l).try_into().unwrap(), 1],
                    other.as_slice(),
                    res.as_mut_slice(),
                )?;
                res.reshape(Shape::Vector(*m as u64))?;
                Ok(res)
            }
            (Shape::Matrix(a, b), Shape::Matrix(c, d)) => {
                let mut res = Self::new_default(Shape::Matrix(*a, *d));
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
            (Shape::Vector(l), shp @ Shape::Tensor(_)) => {
                let [m, n] = shp.last_two().unwrap();

                let it = ColumnIter::new(&other.values, n as usize * m as usize);
                let mut out = Self::new_default([
                    (other.len() / (n as usize * m as usize)) as u32,
                    (*l).try_into().unwrap(),
                ]);
                for (mat, out) in it.zip(ColumnIterMut::new(&mut out.values, *l as usize)) {
                    matmul_impl(
                        [1, (*l).try_into().unwrap()],
                        self.as_slice(),
                        [n, m],
                        mat,
                        out,
                    )?;
                }
                Ok(out)
            }
            (shp @ Shape::Tensor(_), Shape::Vector(l)) => {
                let [m, n] = shp.last_two().unwrap();

                let it = ColumnIter::new(&self.values, n as usize * m as usize);
                let l: u32 = (*l).try_into().unwrap();
                let mut out =
                    Self::new_default([(self.len() / (n as usize * m as usize)) as u32, l]);
                for (mat, out) in it.zip(ColumnIterMut::new(&mut out.values, l as usize)) {
                    matmul_impl([n, m], mat, [l, 1], other.as_slice(), out)?;
                }
                Ok(out)
            }
            (Shape::Matrix(a, b), shp @ Shape::Tensor(_)) => {
                let [a, b] = [*a, *b];
                let [c, d] = shp.last_two().unwrap();

                let it = ColumnIter::new(&other.values, c as usize * d as usize);
                let mut out =
                    Self::new_default(vec![(other.len() / (c as usize * d as usize)) as u32, a, d]);
                for (mat, out) in
                    it.zip(ColumnIterMut::new(&mut out.values, a as usize * d as usize))
                {
                    matmul_impl([a, b], self.as_slice(), [c, d], mat, out)?;
                }
                Ok(out)
            }
            (shp @ Shape::Tensor(_), Shape::Matrix(c, d)) => {
                let [a, b] = shp.last_two().unwrap();
                let [c, d] = [*c, *d];

                let it = ColumnIter::new(&self.values, c as usize * d as usize);
                let mut out =
                    Self::new_default(vec![(self.len() / (c as usize * d as usize)) as u32, a, d]);
                for (mat, out) in
                    it.zip(ColumnIterMut::new(&mut out.values, a as usize * d as usize))
                {
                    matmul_impl([a, b], self.as_slice(), [c, d], mat, out)?;
                }
                Ok(out)
            }
            (ab @ Shape::Tensor(_), cd @ Shape::Tensor(_)) => {
                let [a, b] = ab.last_two().unwrap();
                let [c, d] = cd.last_two().unwrap();

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

                let mut out = Self::new_default(vec![nmatrices as u32, a, d]);

                for (out, (lhs, rhs)) in
                    ColumnIterMut::new(&mut out.values, a as usize * d as usize).zip(it_0.zip(it_1))
                {
                    matmul_impl([a, b], lhs, [c, d], rhs, out)?;
                }

                Ok(out)
            }
        }
    }
}
