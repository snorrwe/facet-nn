//! Matrix operation implementations
//!

use std::ops::{Add, AddAssign, Mul};

use super::{NdArray, NdArrayError};

pub fn matmul<'a, T>(
    [n, m]: [u32; 2],
    values0: &'a [T],
    [m1, p]: [u32; 2],
    values1: &'a [T],
) -> Result<NdArray<T>, NdArrayError>
where
    T: AddAssign + Add<Output = T> + Mul<Output = T> + Default + 'a,
    &'a T: Add<Output = T> + 'a + Mul<Output = T>,
{
    debug_assert_eq!((n as usize * m as usize), values0.len());
    debug_assert_eq!((p as usize * m as usize), values1.len());
    if m != m1 {
        return Err(NdArrayError::DimensionMismatch {
            expected: m as usize,
            actual: m1 as usize,
        });
    }

    let mut res = NdArray::new_default(vec![n as u32, p as u32].into_boxed_slice());
    let output = res.as_mut_slice();

    for i in 0..n {
        for j in 0..p {
            for k in 0..m {
                let val0 = &values0[(i * m + k) as usize];
                let val1 = &values1[(k * p + j) as usize];
                output[(i * p + j) as usize] += val0 * val1
            }
        }
    }

    Ok(res)
}
