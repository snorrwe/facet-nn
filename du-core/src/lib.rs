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
    let res = if shape.len() > 0 {
        ndarray::NdArray::new_with_values(&shape[..shape.len() - 1], res).unwrap()
    } else {
        // scalar
        ndarray::NdArray::new_with_values(0, res).unwrap()
    };
    res
}
