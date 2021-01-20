//! Matrix operation implementations
//!

use std::ops::{Add, AddAssign, Mul};

use super::{
    column_iter::ColumnIter, column_iter::ColumnIterMut, shape::Shape, Data, NdArray, NdArrayError,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Raw matrix multiplication method
///
/// multiplies m*k and k*n matrices and outputs an m*n matrix
// this really should be optimized...
pub fn matmul_impl<'a, T>(
    [m, k, n]: [u32; 3],
    in0: &'a [T],
    in1: &'a [T],
    out: &mut [T],
) -> Result<(), NdArrayError>
where
    T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a + Copy + Send + Sync,
{
    debug_assert_eq!((m as usize * k as usize), in0.len());
    debug_assert_eq!((n as usize * k as usize), in1.len());
    debug_assert_eq!(out.len(), m as usize * n as usize);

    #[cfg(feature = "rayon")]
    {
        let k = k as usize;
        let n = n as usize;
        // iterate over the result's rows
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..row.len() {
                let mut valout = Default::default();
                for l in 0usize..k {
                    let val0 = in0[i * k + l];
                    let val1 = in1[l * n + j];
                    valout += val0 * val1
                }
                row[j] = valout;
            }
        });
    }

    #[cfg(not(feature = "rayon"))]
    for i in 0..m {
        for j in 0..n {
            let mut val = T::default();
            for l in 0..k {
                let val0 = in0[(i * k + l) as usize];
                let val1 = in1[(l * n + j) as usize];
                val += val0 * val1
            }
            out[(i * n + j) as usize] = val;
        }
    }

    Ok(())
}

/// f32 specialized method
pub fn matmul_impl_f32<'a>(
    [m, k, n]: [u32; 3],
    in0: &'a [f32],
    in1: &'a [f32],
    out: &mut [f32],
) -> Result<(), NdArrayError> {
    #[cfg(feature = "gpu")]
    // heuristics determining if we should run on the gpu
    if m * k * n > 1024 && crate::gpu::EXECUTOR.is_some() {
        return match crate::gpu::matmul::matmul_f32_impl([m, k, n], in0, in1, out) {
            Ok(()) => Ok(()),
            Err(crate::gpu::GpuNdArrayError::NdArrayError(err)) => Err(err),
            err @ Err(_) => panic!("{:?}", err),
        };
    }
    matmul_impl([m, k, n], in0, in1, out)
}

pub fn transpose_mat<T: Clone>([m, n]: [usize; 2], inp: &[T], out: &mut [T]) {
    assert!(inp.len() >= m * n);
    assert!(inp.len() <= out.len());
    for (i, col) in ColumnIter::new(inp, n).enumerate() {
        for (j, v) in col.iter().cloned().enumerate() {
            out[j * m + i] = v;
        }
    }
}

pub fn flip_mat_horizontal<T: Clone>([m, n]: [usize; 2], inp: &[T], out: &mut [T]) {
    assert!(inp.len() == m * n);
    assert!(inp.len() <= out.len());

    let mid = m / 2;
    for (i, col) in ColumnIter::new(inp, n).enumerate() {
        for (j, v) in col.iter().cloned().enumerate() {
            let col = (mid + i) % m;
            let ind = col * n + j;
            out[ind] = v;
        }
    }
}

pub fn flip_mat_vertical<T: Clone>([m, n]: [usize; 2], inp: &[T], out: &mut [T]) {
    assert!(inp.len() == m * n);
    assert!(inp.len() <= out.len());

    for (i, col) in ColumnIter::new(inp, n).enumerate() {
        for (j, v) in col.iter().rev().cloned().enumerate() {
            let ind = i * n + j;
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
        let [m, n] = self
            .shape
            .last_two()
            .ok_or_else(|| NdArrayError::UnsupportedShape(self.shape.clone()))?;

        let mut out = Data::new();
        out.resize(self.len(), self.values[0].clone());
        let span = m as usize * n as usize;
        // broadcast tensors as a colection of matrices
        for (i, innermat) in ColumnIter::new(self.as_slice(), span).enumerate() {
            f(
                [m as usize, n as usize],
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
        let [m, n] = self
            .shape
            .last_two()
            .ok_or_else(|| NdArrayError::UnsupportedShape(self.shape.clone()))?;

        if m != n {
            return Err(NdArrayError::DimensionMismatch {
                expected: m as usize,
                actual: n as usize,
            });
        }

        let mut out = Data::new();
        out.resize(self.len(), self.values[0].clone());
        let span = m as usize * n as usize;
        // broadcast tensors as a colection of matrices
        for (i, innermat) in ColumnIter::new(self.as_slice(), span).enumerate() {
            rotate_mat_cw(m as usize, innermat, &mut out.as_mut_slice()[i * span..]);
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

            (Shape::Vector([l]), Shape::Matrix([_, n])) => {
                out.reshape(Shape::Matrix([1, *n]));
                f(
                    [1, *l, *n],
                    self.as_slice(),
                    other.as_slice(),
                    out.as_mut_slice(),
                )?;
                out.reshape(*n);
                Ok(())
            }
            (Shape::Matrix([m, n]), Shape::Vector([_])) => {
                out.reshape(Shape::Matrix([*m, 1]));
                f(
                    [*m, *n, 1],
                    self.as_slice(),
                    other.as_slice(),
                    out.as_mut_slice(),
                )?;
                out.reshape(*n);
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
                let [n, m] = shp.last_two().unwrap();

                let it = ColumnIter::new(&other.values, m as usize * n as usize);
                out.reshape([(other.len() / (m as usize * n as usize)) as u32, *l]);
                for (mat, out) in it.zip(ColumnIterMut::new(&mut out.values, *l as usize)) {
                    f([1, *l, n], self.as_slice(), mat, out)?;
                }
                Ok(())
            }
            (shp @ Shape::Tensor(_), Shape::Vector([l])) => {
                let [n, m] = shp.last_two().unwrap();

                let it = ColumnIter::new(&self.values, m as usize * n as usize);
                let l: u32 = *l;
                out.reshape([(self.len() / (m as usize * n as usize)) as u32, l]);
                for (mat, out) in it.zip(ColumnIterMut::new(&mut out.values, l as usize)) {
                    f([m, n, 1], mat, other.as_slice(), out)?;
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
