//!
//! ## Features
//!
//! - `rayon`: Enables CPU parallelism via the rayon crate
//!
use ndarray::{shape::Shape, NdArrayError};

pub mod activation;
pub mod layer;
pub mod loss;
pub mod ndarray;
pub mod prelude;

#[cfg(test)]
mod tests;

#[cfg(feature = "rayon")]
pub use rayon;
pub use smallvec;

pub type DuResult<T> = Result<T, DuError>;

#[derive(Debug, thiserror::Error)]
pub enum DuError {
    #[error("Error in an NdArray: {0}")]
    ArrayError(NdArrayError),
    #[error("Binary function got mismatching shapes {0:?} {1:?}")]
    MismatchedShapes(Shape, Shape),
}

impl From<NdArrayError> for DuError {
    fn from(err: NdArrayError) -> Self {
        DuError::ArrayError(err)
    }
}

pub trait SquareRoot {
    fn sqrt(self) -> Self;
}

impl SquareRoot for f32 {
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl SquareRoot for f64 {
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

pub fn sum<'a, T>(inp: &ndarray::NdArray<T>) -> ndarray::NdArray<T>
where
    T: 'a + std::iter::Sum + Copy,
{
    let res = inp
        .iter_rows()
        .map(|x| x.iter().cloned().sum())
        .collect::<ndarray::Data<_>>();

    let shape = inp.shape();
    let shape = shape.as_slice();
    if !shape.is_empty() {
        ndarray::NdArray::new_with_values(&shape[..shape.len() - 1], res).unwrap()
    } else {
        // scalar
        ndarray::NdArray::new_with_values(0, res).unwrap()
    }
}

/// Calculate the column-wise mean.
///
/// Scalars will return themselves. While others will collapse the last column into a 1D vector.
///
/// To calculate the mean of all items reshape the array into a vector first.
///
///
/// ## Mean of tensor
///
/// ```
/// use facet_core::prelude::*;
///
/// let a = NdArray::new_with_values(&[2u32, 2, 3][..], smallvec![
///  1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32,
///  12.0f32
/// ]).unwrap();
///
/// let m = mean(&a).unwrap();
///
/// assert_eq!(a.shape(), &Shape::from(&[2, 2, 3][..]));
/// assert_eq!(m.shape(), &Shape::from(&[2, 2, 1][..]));
/// assert_eq!(m.as_slice(), &[2.0, 5.0, 8.0, 11.0]);
///
/// ```
///
/// ## Mean of vector
///
/// ```
/// use facet_core::prelude::*;
///
/// let a = NdArray::new_vector(smallvec![ 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0 ]);
///
/// let m = mean(&a).unwrap();
///
/// assert_eq!(a.shape(), &Shape::from(&[12][..]));
/// assert_eq!(m.shape(), &Shape::from(&[0][..]));
/// assert_eq!(m.as_slice(), &[6.5]);
///
/// ```
pub fn mean<T>(inp: &ndarray::NdArray<T>) -> Result<ndarray::NdArray<T>, NdArrayError>
where
    T: Copy + Default + std::iter::Sum + std::ops::Div<f32, Output = T>,
{
    match inp.shape() {
        Shape::Scalar(_) => Ok(inp.clone()),
        Shape::Vector([n]) => {
            let s: T = inp.as_slice().iter().cloned().sum();
            let res = s / *n as f32;
            let mut values = ndarray::Data::new();
            values.push(res);
            ndarray::NdArray::new_with_values(0, values)
        }
        Shape::Tensor(_) | Shape::Matrix([_, _]) => {
            let mut values = Vec::with_capacity(inp.shape().col_span());
            for col in inp.iter_rows() {
                let s: T = col.iter().cloned().sum();
                let res = s / (col.len() as f32);
                values.push(res)
            }
            let mut res = ndarray::NdArray::new_vector(values);
            let mut shape = inp.shape().clone();
            let l = shape.as_slice().len();
            shape.as_mut_slice()[l - 1] = 1;
            res.reshape(shape);
            Ok(res)
        }
    }
}

/// Calculate moving averages of the innermost dimension.
///
/// For matrices and tensors return a matrix where each columns will contain the moving averages of
/// their corresponding columns.
///
/// ```
/// use facet_core::prelude::*;
///
/// let values = NdArray::new_vector( vec![9.0, 8.0,9.0,12.0,9.,12.,11.,7.,13.,9.,11.,10.] );
/// let ma = facet_core::moving_average(&values, 3).unwrap();
///
/// assert_eq!(ma.as_slice(),
/// &[  8.666666666666666, 9.666666666666666, 10.0, 11.0,
///     10.666666666666666, 10.0, 10.333333333333334,
///     9.666666666666666, 11.0, 10.0]);
/// ```
pub fn moving_average<T>(
    inp: &ndarray::NdArray<T>,
    window: usize,
) -> Result<ndarray::NdArray<T>, NdArrayError>
where
    T: Copy + Default + std::iter::Sum + std::ops::Div<f32, Output = T> + std::fmt::Debug,
{
    use crate::prelude::*;
    if window == 0 {
        return Err(NdArrayError::BadInput(
            "window must be non-negative".to_string(),
        ));
    }
    let winf = f64::from(window as u32) as f32;
    let arr = match inp.shape() {
        Shape::Scalar(_) => return Err(NdArrayError::UnsupportedShape(inp.shape().clone())),
        Shape::Vector(_) => {
            let data = inp
                .as_slice()
                .windows(window)
                .map(|window| {
                    let sum: T = window.iter().cloned().sum();
                    sum / winf
                })
                .fold(prelude::Data::default(), |mut res, mean| {
                    res.push(mean);
                    res
                });

            ndarray::NdArray::new_vector(data)
        }
        _ => {
            let n: u32 = inp.shape().last();
            let win_sum = inp
                .as_slice()
                .windows(window * n as usize)
                .step_by(n as usize)
                .fold(Data::default(), |mut res, x| {
                    for i in 0..n {
                        let s: T = x.iter().skip(i as usize).step_by(n as usize).cloned().sum();
                        let mean = s / winf;
                        res.push(mean);
                    }
                    res
                });

            ndarray::NdArray::new_with_values([(win_sum.len() / n as usize) as u32, n], win_sum)
                .unwrap()
        }
    };

    Ok(arr)
}

/// Calculate the standard deviation for each column in the array.
///
/// If the input shape is a scalar return Default::default() (0.0 for numeric types).
/// While other shapes will collapse the last column into a 1D vector.
///
/// Otherwise return the std for each item in the array.
///
///
/// ```
/// use facet_core::ndarray::NdArray;
/// let a = NdArray::new_vector(
///     vec![9.0, 2.0, 5.0, 4.0, 12.0, 7.0, 8.0, 11.0, 9.0, 3.0, 7.0, 4.0, 12.0, 5.0, 4.0, 10.0, 9.0, 6.0, 9.0, 4.0]
/// );
///
/// let b = facet_core::std(&a, None).expect("Failed to perform std");
///
/// assert_eq!(b.as_slice(), &[2.9832866]);
/// ```
pub fn std<'a, T>(
    inp: &'a ndarray::NdArray<T>,
    mean: Option<&'a ndarray::NdArray<T>>,
) -> Result<ndarray::NdArray<T>, NdArrayError>
where
    T: Copy
        + Default
        + std::iter::Sum
        + std::ops::Div<f32, Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<T, Output = T>
        + SquareRoot,
{
    let mut stdsq = std_squared(inp, mean)?;
    stdsq
        .as_mut_slice()
        .iter_mut()
        .for_each(|x| *x = x.clone().sqrt());
    Ok(stdsq)
}

/// Calculate square of the standard deviation for each column in the array.
///
/// If the input shape is a scalar return Default::default() (0.0 for numeric types).
/// While other shapes will collapse the last column into a 1D vector.
///
/// Otherwise return the std for each item in the array.
///
///
/// ```
/// use facet_core::ndarray::NdArray;
/// let a = NdArray::new_vector(
///     vec![9.0, 2.0, 5.0, 4.0, 12.0, 7.0, 8.0, 11.0, 9.0, 3.0, 7.0, 4.0, 12.0, 5.0, 4.0, 10.0, 9.0, 6.0, 9.0, 4.0]
/// );
///
/// let b = facet_core::std_squared(&a, None).expect("Failed to perform std");
///
/// assert_eq!(b.as_slice(), &[8.9]);
/// ```
pub fn std_squared<'a, T>(
    inp: &'a ndarray::NdArray<T>,
    mean: Option<&'a ndarray::NdArray<T>>,
) -> Result<ndarray::NdArray<T>, NdArrayError>
where
    T: Copy
        + Default
        + std::iter::Sum
        + std::ops::Mul<Output = T>
        + std::ops::Div<f32, Output = T>
        + std::ops::Sub<T, Output = T>,
{
    if matches!(inp.shape(), Shape::Scalar(_)) {
        return Ok(ndarray::NdArray::new_default(inp.shape().clone()));
    }

    match mean {
        Some(mean) => _std_squared_from_mean(inp, mean),
        None => {
            let mean = crate::mean(inp)?;
            _std_squared_from_mean(inp, &mean)
        }
    }
}

fn _std_squared_from_mean<'a, T>(
    inp: &'a ndarray::NdArray<T>,
    mean: &'a ndarray::NdArray<T>,
) -> Result<ndarray::NdArray<T>, NdArrayError>
where
    T: Copy
        + std::ops::Mul<Output = T>
        + std::iter::Sum
        + std::ops::Div<f32, Output = T>
        + Default
        + std::ops::Sub<T, Output = T>,
{
    let res = inp
        .iter_rows()
        .zip(mean.iter_rows().map(|mean| {
            debug_assert_eq!(mean.len(), 1);
            mean[0]
        }))
        .map(move |(col, m)| {
            let s: T = col
                .iter()
                .map(move |x| {
                    let d: T = *x - m;
                    d * d
                })
                .sum();

            s / col.len() as f32
        })
        .collect::<prelude::Data<T>>();

    let mut shape = inp.shape().clone();
    let l = shape.as_slice().len();
    shape.as_mut_slice()[l - 1] = 1;
    ndarray::NdArray::new_with_values(shape.clone(), res)
}

pub fn clip<T>(inp: &mut ndarray::NdArray<T>, min: T, max: T)
where
    T: Copy + std::cmp::PartialOrd,
{
    use std::cmp::Ordering;
    inp.as_mut_slice().iter_mut().for_each(|x| {
        if let Some(Ordering::Less) = (*x).partial_cmp(&min) {
            *x = min
        } else if let Some(Ordering::Greater) = (*x).partial_cmp(&max) {
            *x = max
        }
    })
}

/// calculate `y=1/sqrt(x)` for each x element in the input array
pub fn fast_inv_sqrt_f32(inp: &ndarray::NdArray<f32>, out: &mut ndarray::NdArray<f32>) {
    out.reshape(inp.shape().clone());
    let len = inp.as_slice().len();
    for i in 0..len {
        out.as_mut_slice()[i] = _fast_inv_sqrt_f32(inp.as_slice()[i]);
    }
}

/// Normalize the inner column vectors using [fast_inv_sqrt](https://en.wikipedia.org/wiki/Fast_inverse_square_root) algorithm. Meaning slightly inaccurate
/// results. E.g. a vector might have a length slightly less than 1.
pub fn normalize_f32_vectors(inp: &ndarray::NdArray<f32>, out: &mut ndarray::NdArray<f32>) {
    out.reshape(inp.shape().clone());
    let collen = inp.shape().last() as usize;
    for (inp, out) in inp.iter_rows().zip(out.iter_rows_mut()) {
        let mut vec_len = 0.0;
        for i in inp.iter().take(collen) {
            vec_len += i * i;
        }
        let inv_vec_len = _fast_inv_sqrt_f32(vec_len);

        for i in 0..inp.len() {
            out[i] = inp[i] * inv_vec_len;
        }
    }
}

#[inline]
fn _fast_inv_sqrt_f32(mut y: f32) -> f32 {
    const THREE_HALVES: f32 = 1.5;
    let x2 = y * 0.5;

    unsafe {
        union F {
            i: i32,
            f: f32,
        }

        let mut f: F = F { f: y };
        f.i = 0x5F375A86 - (f.i >> 1);
        y = f.f;
    }

    // Newton method, 1 iteration is enough to achive ~1% error
    y = y * (THREE_HALVES - (x2 * y * y));
    y
}

/// calculate `y=1/sqrt(x)` for each x element in the input array
pub fn fast_inv_sqrt_f64(inp: &ndarray::NdArray<f64>, out: &mut ndarray::NdArray<f64>) {
    out.reshape(inp.shape().clone());
    let len = inp.as_slice().len();
    for i in 0..len {
        out.as_mut_slice()[i] = _fast_inv_sqrt_f64(inp.as_slice()[i]);
    }
}

/// Normalize the inner column vectors using [fast_inv_sqrt](https://en.wikipedia.org/wiki/Fast_inverse_square_root) algorithm. Meaning slightly inaccurate
/// results. E.g. a vector might have a length slightly less than 1.
pub fn normalize_f64_vectors(inp: &ndarray::NdArray<f64>, out: &mut ndarray::NdArray<f64>) {
    out.reshape(inp.shape().clone());
    let collen = inp.shape().last() as usize;
    for (inp, out) in inp.iter_rows().zip(out.iter_rows_mut()) {
        let mut vec_len = 0.0;
        for i in inp.iter().take(collen) {
            vec_len += i * i;
        }
        let inv_vec_len = _fast_inv_sqrt_f64(vec_len);

        for i in 0..inp.len() {
            out[i] = inp[i] * inv_vec_len;
        }
    }
}

#[inline]
fn _fast_inv_sqrt_f64(mut y: f64) -> f64 {
    const THREE_HALVES: f64 = 1.5;
    let x2 = y * 0.5;

    unsafe {
        union F {
            i: i64,
            f: f64,
        }

        let mut f: F = F { f: y };
        f.i = 0x5FE6EB50C7B537A9 - (f.i >> 1);
        y = f.f;
    }

    // Newton method, 1 iteration is enough to achive ~1% error
    y = y * (THREE_HALVES - (x2 * y * y));
    y
}

/// Calculate the square of the eucledian length of the vector(s). Matrices and Tensors are treated as a list of vectors.
pub fn veclen_squared<'a, T>(inp: &'a ndarray::NdArray<T>, out: &mut ndarray::NdArray<T>)
where
    T: Copy + Default + std::ops::AddAssign,
    &'a T: std::ops::Mul<Output = T>,
{
    if matches!(inp.shape(), Shape::Scalar(_)) {
        *out = inp.clone();
    }
    let shape_len = inp.shape().as_slice().len();
    let out_shape = &inp.shape().as_slice()[..shape_len - 1];
    out.reshape(out_shape);
    for (i, vector) in inp.iter_rows().enumerate() {
        let mut vec_len = T::default();
        for x in vector {
            vec_len += x * x;
        }
        out.as_mut_slice()[i] = vec_len;
    }
}

/// Calculate the eucledian length of the vector(s). Matrices and Tensors are treated as a list of vectors.
pub fn veclen<'a, T>(inp: &'a ndarray::NdArray<T>, out: &mut ndarray::NdArray<T>)
where
    T: Copy + Default + std::ops::AddAssign + SquareRoot,
    &'a T: std::ops::Mul<Output = T>,
{
    veclen_squared(inp, out);
    for x in out.as_mut_slice() {
        *x = x.sqrt();
    }
}
