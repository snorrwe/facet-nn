use ndarray::{shape::Shape, NdArrayError};

pub mod activation;
pub mod layer;
pub mod loss;
pub mod ndarray;

pub type DuResult<T> = Result<T, DuError>;

#[cfg(feature = "rayon")]
pub use rayon;
pub use smallvec;

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

pub fn sum<'a, T>(inp: &ndarray::NdArray<T>) -> ndarray::NdArray<T>
where
    T: 'a + std::iter::Sum + Copy,
{
    let res = inp
        .iter_cols()
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

pub fn mean<T>(inp: &ndarray::NdArray<T>) -> Result<ndarray::NdArray<T>, NdArrayError>
where
    T: Clone + Default + std::iter::Sum + std::ops::Div<Output = T> + std::convert::TryFrom<u32>,
{
    match inp.shape() {
        Shape::Scalar(_) => Ok(inp.clone()),
        Shape::Vector([n]) => {
            let s: T = inp.as_slice().iter().cloned().sum();
            let res = s / T::try_from(*n)
                .map_err(|_| NdArrayError::ConversionError(format!("{:?}", n)))?;
            let mut values = ndarray::Data::new();
            values.push(res);
            ndarray::NdArray::new_with_values(0, values)
        }
        Shape::Tensor(_) | Shape::Matrix([_, _]) => {
            let mut values = Vec::with_capacity(inp.shape().col_span());
            for col in inp.iter_cols() {
                let s: T = col.iter().cloned().sum();
                let res = s
                    / (T::try_from(col.len() as u32))
                        .map_err(|_| NdArrayError::ConversionError(format!("{:?}", col.len())))?;
                values.push(res)
            }
            let mut res = ndarray::NdArray::new_vector(values);
            let shape = inp.shape().as_slice();
            res.reshape(&shape[..shape.len() - 1]);
            Ok(res)
        }
    }
}
