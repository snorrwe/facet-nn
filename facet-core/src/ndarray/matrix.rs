//! Matrix operation implementations
//!

use std::ops::{Add, AddAssign, Mul};

use super::{
    column_iter::ColumnIter, column_iter::ColumnIterMut, shape::Shape, Data, NdArray, NdArrayError,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Raw matrix multiplication method
// this really should be optimized further...
pub fn matmul_impl<'a, T>(
    [n, m]: [u32; 2],
    values0: &'a [T],
    [m1, p]: [u32; 2],
    values1: &'a [T],
    out: &mut [T],
) -> Result<(), NdArrayError>
where
    T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a + Copy + Send + Sync,
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

    #[cfg(feature = "rayon")]
    {
        let m = m as usize;
        let p = p as usize;
        // iterate over the result's columns
        out.par_chunks_mut(p).enumerate().for_each(|(i, out)| {
            out.par_iter_mut().enumerate().for_each(|(j, valout)| {
                *valout = Default::default();
                for k in 0usize..m {
                    let val0 = &values0[i * m + k];
                    let val1 = &values1[k * p + j];
                    *valout += *val0 * *val1
                }
            });
        });
    }

    #[cfg(not(feature = "rayon"))]
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
    assert!(inp.len() == n * m);
    assert!(inp.len() <= out.len());
    for (i, col) in ColumnIter::new(inp, m).enumerate() {
        for (j, v) in col.iter().cloned().enumerate() {
            out[j * n + i] = v;
        }
    }
}

pub fn flip_mat_horizontal<T: Clone>([n, m]: [usize; 2], inp: &[T], out: &mut [T]) {
    assert!(inp.len() == n * m);
    assert!(inp.len() <= out.len());

    let mid = n / 2;
    for (i, col) in ColumnIter::new(inp, m).enumerate() {
        for (j, v) in col.iter().cloned().enumerate() {
            let col = (mid + i) % n;
            let ind = col * m + j;
            out[ind] = v;
        }
    }
}

pub fn flip_mat_vertical<T: Clone>([n, m]: [usize; 2], inp: &[T], out: &mut [T]) {
    assert!(inp.len() == n * m);
    assert!(inp.len() <= out.len());

    for (i, col) in ColumnIter::new(inp, m).enumerate() {
        for (j, v) in col.iter().rev().cloned().enumerate() {
            let ind = i * m + j;
            out[ind] = v;
        }
    }
}

/// rotates all elements clockwise in a square matrix
///
///
/// ```txt
/// | a  b |
/// | c  d |
///
/// // becomes
/// | c  a |
/// | d  b |
/// ```
pub fn rotate_mat_cw<T: Clone>(col: usize, inp: &[T], out: &mut [T]) {
    let mut intermediate: smallvec::SmallVec<[T; 32]> =
        smallvec::smallvec![inp[0].clone(); inp.len()];
    transpose_mat([col, col], inp, intermediate.as_mut_slice());
    flip_mat_horizontal([col, col], intermediate.as_slice(), out);
}

impl<T> NdArray<T> {
    /// Tensors are broadcast as a list of matrices
    pub fn flip_mat_vertical(&self) -> Result<Self, NdArrayError>
    where
        T: Clone,
    {
        self.flip_impl(flip_mat_vertical)
    }

    /// Tensors are broadcast as a list of matrices
    pub fn flip_mat_horizontal(&self) -> Result<Self, NdArrayError>
    where
        T: Clone,
    {
        self.flip_impl(flip_mat_horizontal)
    }

    fn flip_impl(&self, f: fn([usize; 2], &[T], &mut [T])) -> Result<Self, NdArrayError>
    where
        T: Clone,
    {
        let [n, m] = self
            .shape
            .last_two()
            .ok_or_else(|| NdArrayError::UnsupportedShape(self.shape.clone()))?;

        let mut out = Data::new();
        out.resize(self.len(), self.values[0].clone());
        let span = n as usize * m as usize;
        // broadcast tensors as a colection of matrices
        for (i, innermat) in ColumnIter::new(self.as_slice(), span).enumerate() {
            f(
                [n as usize, m as usize],
                innermat,
                &mut out.as_mut_slice()[i * span..],
            );
        }

        Self::new_with_values(self.shape.clone(), out)
    }

    /// Tensors are broadcast as a list of matrices
    ///
    /// Rotates the contents of invidual square matrices clockwise.
    ///
    /// ```
    /// use facet_core::ndarray::NdArray;
    ///
    /// let inp = NdArray::new_with_values(
    ///     &[3, 2, 2][..],
    ///     // 3x matrix
    ///     // | 1  3 |
    ///     // | 2  4 |
    ///     vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4].into(),
    /// ).unwrap();
    ///
    /// let out = inp.rotate_cw().unwrap();
    ///
    /// assert_eq!(out.as_slice(),
    /// // 3x matrix
    /// // | 2  1 |
    /// // | 4  3 |
    /// &[
    ///     2, 4, 1, 3,
    ///     2, 4, 1, 3,
    ///     2, 4, 1, 3,
    /// ]
    /// );
    /// ```
    pub fn rotate_cw(&self) -> Result<Self, NdArrayError>
    where
        T: Clone,
    {
        let [n, m] = self
            .shape
            .last_two()
            .ok_or_else(|| NdArrayError::UnsupportedShape(self.shape.clone()))?;

        if n != m {
            return Err(NdArrayError::DimensionMismatch {
                expected: n as usize,
                actual: m as usize,
            });
        }

        let mut out = Data::new();
        out.resize(self.len(), self.values[0].clone());
        let span = n as usize * m as usize;
        // broadcast tensors as a colection of matrices
        for (i, innermat) in ColumnIter::new(self.as_slice(), span).enumerate() {
            rotate_mat_cw(n as usize, innermat, &mut out.as_mut_slice()[i * span..]);
        }

        let mut shape = self.shape.clone();
        // swap m, n because we did a transpose
        let s = shape.as_mut_slice();
        let l = s.len();
        debug_assert!(l >= 2);
        s[l - 1] = n;
        s[l - 2] = m;

        Self::new_with_values(shape, out)
    }

    /// - Scalars not allowed.
    /// - Tensor arrays are treated as a colection of matrices and are broadcast accordingly
    ///
    /// ## Tensor arrays Example
    ///
    /// This example will take a single 2 by 3 matrix and multiply it with 2 3 by 2 matrices.
    /// The output is 2(!) 2 by 2 matrices.
    ///
    ///
    /// ```
    /// use facet_core::ndarray::{NdArray, Data};
    /// use facet_core::ndarray::shape::Shape;
    ///
    /// // 2 by 3 matrix
    /// let a = NdArray::new_with_values([2, 3], Data::from_slice(&[1, 2, -1, 2, 0, 1])).unwrap();
    ///
    /// // the same 3 by 2 matrix twice
    /// let b = NdArray::new_with_values(
    ///     &[2, 3, 2][..],
    ///     Data::from_slice(&[3, 1, 0, -1, -2, 3, /*|*/ 3, 1, 0, -1, -2, 3]),
    /// )
    /// .unwrap();
    ///
    /// let mut c = NdArray::new(0);
    /// a.matmul(&b, &mut c).expect("matmul");
    ///
    /// // output 2 2 by 2 matrices
    /// assert_eq!(c.shape(), &Shape::Tensor((&[2, 2, 2][..]).into()));
    /// assert_eq!(c.as_slice(), &[5, -4, 4, 5, /*|*/ 5, -4, 4, 5]);
    /// ```
    pub fn matmul<'a>(&'a self, other: &'a Self, out: &mut Self) -> Result<(), NdArrayError>
    where
        T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a + Copy + Sync + Send,
    {
        match (&self.shape, &other.shape) {
            shapes @ (Shape::Scalar(_), Shape::Scalar(_))
            | shapes @ (Shape::Scalar(_), Shape::Vector(_))
            | shapes @ (Shape::Scalar(_), Shape::Matrix(_))
            | shapes @ (Shape::Scalar(_), Shape::Tensor(_))
            | shapes @ (Shape::Vector(_), Shape::Scalar(_))
            | shapes @ (Shape::Matrix(_), Shape::Scalar(_))
            | shapes @ (Shape::Tensor(_), Shape::Scalar(_)) => {
                Err(NdArrayError::BinaryOpNotSupported {
                    shape_a: shapes.0.clone(),
                    shape_b: shapes.1.clone(),
                })
            }

            (Shape::Vector([a]), Shape::Vector([b])) => {
                let res = self.inner(other).ok_or(NdArrayError::DimensionMismatch {
                    expected: *a as usize,
                    actual: *b as usize,
                })?;
                *out = Self::new_with_values(&[][..], Data::from_slice(&[res][..]))?;
                Ok(())
            }

            (Shape::Vector([l]), Shape::Matrix([n, m])) => {
                out.reshape(Shape::Matrix([1, *m]));
                matmul_impl(
                    [1, *l],
                    self.as_slice(),
                    [*n, *m],
                    other.as_slice(),
                    out.as_mut_slice(),
                )?;
                out.reshape(*m);
                Ok(())
            }
            (Shape::Matrix([n, m]), Shape::Vector([l])) => {
                out.reshape(Shape::Matrix([*n, 1]));
                matmul_impl(
                    [*n, *m],
                    self.as_slice(),
                    [*l, 1],
                    other.as_slice(),
                    out.as_mut_slice(),
                )?;
                out.reshape(*m);
                Ok(())
            }
            (Shape::Matrix([a, b]), Shape::Matrix([c, d])) => {
                out.reshape(Shape::Matrix([*a, *d]));
                matmul_impl(
                    [*a, *b],
                    self.as_slice(),
                    [*c, *d],
                    other.as_slice(),
                    out.as_mut_slice(),
                )?;
                Ok(())
            }

            // broadcast matrices
            (Shape::Vector([l]), shp @ Shape::Tensor(_)) => {
                let [m, n] = shp.last_two().unwrap();

                let it = ColumnIter::new(&other.values, n as usize * m as usize);
                out.reshape([(other.len() / (n as usize * m as usize)) as u32, *l]);
                for (mat, out) in it.zip(ColumnIterMut::new(&mut out.values, *l as usize)) {
                    matmul_impl([1, *l], self.as_slice(), [n, m], mat, out)?;
                }
                Ok(())
            }
            (shp @ Shape::Tensor(_), Shape::Vector([l])) => {
                let [m, n] = shp.last_two().unwrap();

                let it = ColumnIter::new(&self.values, n as usize * m as usize);
                let l: u32 = *l;
                out.reshape([(self.len() / (n as usize * m as usize)) as u32, l]);
                for (mat, out) in it.zip(ColumnIterMut::new(&mut out.values, l as usize)) {
                    matmul_impl([n, m], mat, [l, 1], other.as_slice(), out)?;
                }
                Ok(())
            }
            (Shape::Matrix([a, b]), shp @ Shape::Tensor(_)) => {
                let [a, b] = [*a, *b];
                let [c, d] = shp.last_two().unwrap();

                let it = ColumnIter::new(&other.values, c as usize * d as usize);
                out.reshape(vec![(other.len() / (c as usize * d as usize)) as u32, a, d]);
                for (mat, out) in
                    it.zip(ColumnIterMut::new(&mut out.values, a as usize * d as usize))
                {
                    matmul_impl([a, b], self.as_slice(), [c, d], mat, out)?;
                }
                Ok(())
            }
            (shp @ Shape::Tensor(_), Shape::Matrix([c, d])) => {
                let [a, b] = shp.last_two().unwrap();
                let [c, d] = [*c, *d];

                let it = ColumnIter::new(&self.values, c as usize * d as usize);
                out.reshape(vec![(self.len() / (c as usize * d as usize)) as u32, a, d]);
                for (mat, out) in
                    it.zip(ColumnIterMut::new(&mut out.values, a as usize * d as usize))
                {
                    matmul_impl([a, b], self.as_slice(), [c, d], mat, out)?;
                }
                Ok(())
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

                *out = Self::new_default(vec![nmatrices as u32, a, d]);

                #[cfg(not(feature = "rayon"))]
                {
                    for (out, (lhs, rhs)) in
                        ColumnIterMut::new(&mut out.values, a as usize * d as usize)
                            .zip(it_0.zip(it_1))
                    {
                        matmul_impl([a, b], lhs, [c, d], rhs, out)?;
                    }
                }
                #[cfg(feature = "rayon")]
                {
                    let it_0 = self.values.as_slice().par_chunks(a as usize * b as usize);
                    let it_1 = other.values.as_slice().par_chunks(a as usize * b as usize);

                    out.values
                        .as_mut_slice()
                        .par_chunks_mut(a as usize * d as usize)
                        .zip(it_0.zip(it_1))
                        .try_for_each(|(out, (lhs, rhs))| {
                            matmul_impl([a, b], lhs, [c, d], rhs, out)
                        })?;
                }

                Ok(())
            }
        }
    }
}
