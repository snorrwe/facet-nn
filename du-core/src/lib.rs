use ndarray::{shape::Shape, NdArrayError};

pub mod activation;
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
