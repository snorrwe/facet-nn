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
    [n, m, p]: [u32; 3],
    values0: &'a [T],
    values1: &'a [T],
    out: &mut [T],
) -> Result<(), NdArrayError>
where
    T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a + Copy + Send + Sync,
{
    debug_assert_eq!((n as usize * m as usize), values0.len());
    debug_assert_eq!((p as usize * m as usize), values1.len());
    debug_assert_eq!(out.len(), n as usize * p as usize);

    #[cfg(feature = "rayon")]
    {
        let m = m as usize;
        let p = p as usize;
        // iterate over the result's rows
        out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            for j in 0..row.len() {
                let mut valout = Default::default();
                for k in 0usize..m {
                    let val0 = values0[i * m + k];
                    let val1 = values1[k * p + j];
                    valout += val0 * val1
                }
                row[j] = valout;
            }
        });
    }

    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
        for j in 0..p {
            let mut val = T::default();
            for k in 0..m {
                let val0 = values0[(i * m + k) as usize];
                let val1 = values1[(k * p + j) as usize];
                val += val0 * val1
            }
            out[(i * p + j) as usize] = val;
        }
    }

    Ok(())
}

/// f32 specialized method
pub fn matmul_impl_f32<'a>(
    [n, m, p]: [u32; 3],
    values0: &'a [f32],
    values1: &'a [f32],
    out: &mut [f32],
) -> Result<(), NdArrayError> {
    #[cfg(feature = "gpu")]
    if n >= crate::gpu::matmul::LOCAL_SIZE_X || p >= crate::gpu::matmul::LOCAL_SIZE_Y {
        return match crate::gpu::matmul::matmul_f32_impl([n, m, p], values0, values1, out) {
            Ok(()) => Ok(()),
            Err(crate::gpu::GpuNdArrayError::NdArrayError(err)) => Err(err),
            err @ Err(_) => panic!("{:?}", err),
        };
    }
    matmul_impl([n, m, p], values0, values1, out)
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

impl NdArray<f32> {
    /// specialized matmul
    pub fn matmul_f32<'a>(&'a self, other: &'a Self, out: &mut Self) -> Result<(), NdArrayError> {
        self._matmul(other, out, matmul_impl_f32)
    }
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

        let shape = self.shape.clone();
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
        self._matmul(other, out, matmul_impl)
    }

    fn _matmul<'a, F>(&'a self, other: &'a Self, out: &mut Self, f: F) -> Result<(), NdArrayError>
    where
        F: Fn([u32; 3], &'a [T], &'a [T], &mut [T]) -> Result<(), NdArrayError> + Sync,
        T: Default + Send + Sync,
    {
        match (&self.shape, &other.shape) {
            shapes @ (Shape::Scalar(_), Shape::Scalar(_))
            | shapes @ (Shape::Scalar(_), Shape::Vector(_))
            | shapes @ (Shape::Scalar(_), Shape::Matrix(_))
            | shapes @ (Shape::Scalar(_), Shape::Tensor(_))
            | shapes @ (Shape::Vector(_), Shape::Scalar(_))
            | shapes @ (Shape::Matrix(_), Shape::Scalar(_))
            | shapes @ (Shape::Tensor(_), Shape::Scalar(_))
            | shapes @ (Shape::Vector(_), Shape::Vector(_)) => {
                Err(NdArrayError::BinaryOpNotSupported {
                    shape_a: shapes.0.clone(),
                    shape_b: shapes.1.clone(),
                })
            }

            (Shape::Vector([l]), Shape::Matrix([_, m])) => {
                out.reshape(Shape::Matrix([1, *m]));
                f(
                    [1, *l, *m],
                    self.as_slice(),
                    other.as_slice(),
                    out.as_mut_slice(),
                )?;
                out.reshape(*m);
                Ok(())
            }
            (Shape::Matrix([n, m]), Shape::Vector([_])) => {
                out.reshape(Shape::Matrix([*n, 1]));
                f(
                    [*n, *m, 1],
                    self.as_slice(),
                    other.as_slice(),
                    out.as_mut_slice(),
                )?;
                out.reshape(*m);
                Ok(())
            }
            (Shape::Matrix([a, b]), Shape::Matrix([_, d])) => {
                out.reshape(Shape::Matrix([*a, *d]));
                f(
                    [*a, *b, *d],
                    self.as_slice(),
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
                    f([1, *l, m], self.as_slice(), mat, out)?;
                }
                Ok(())
            }
            (shp @ Shape::Tensor(_), Shape::Vector([l])) => {
                let [m, n] = shp.last_two().unwrap();

                let it = ColumnIter::new(&self.values, n as usize * m as usize);
                let l: u32 = *l;
                out.reshape([(self.len() / (n as usize * m as usize)) as u32, l]);
                for (mat, out) in it.zip(ColumnIterMut::new(&mut out.values, l as usize)) {
                    f([n, m, 1], mat, other.as_slice(), out)?;
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
                    f([a, b, d], self.as_slice(), mat, out)?;
                }
                Ok(())
            }
            (shp @ Shape::Tensor(_), Shape::Matrix([c, d])) => {
                let [a, b] = shp.last_two().unwrap();
                let [c, d] = [*c, *d];

                assert_eq!(b, c);

                let it = ColumnIter::new(&self.values, a as usize * b as usize);
                out.reshape(vec![(self.len() / (c as usize * d as usize)) as u32, a, d]);
                for (mat, out) in
                    it.zip(ColumnIterMut::new(&mut out.values, a as usize * d as usize))
                {
                    f([a, b, d], self.as_slice(), mat, out)?;
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
                    let it_0 = self.values.as_slice().chunks(a as usize * b as usize);
                    let it_1 = other.values.as_slice().chunks(a as usize * b as usize);
                    for (out, (lhs, rhs)) in
                        ColumnIterMut::new(&mut out.values, a as usize * d as usize)
                            .zip(it_0.zip(it_1))
                    {
                        f([a, b, d], lhs, rhs, out)?;
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
                        .try_for_each(|(out, (lhs, rhs))| f([a, b, d], lhs, rhs, out))?;
                }

                Ok(())
            }
        }
    }
}
