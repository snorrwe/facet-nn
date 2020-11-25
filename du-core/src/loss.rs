use std::f64::consts::E;

use crate::{ndarray::NdArray, DuError, DuResult};

/// The author recommends running `softmax` on the output before calling this function
///
/// Expect the predictions to be in the interval [0, 1]
pub fn categorical_cross_entropy(
    predictions: &NdArray<f64>,
    targets: &NdArray<f64>,
) -> DuResult<NdArray<f64>> {
    if predictions.shape() != targets.shape() {
        return Err(DuError::MismatchedShapes(
            predictions.shape().clone(),
            targets.shape().clone(),
        ));
    }

    let mut out = Vec::with_capacity(
        predictions.shape().span() / predictions.shape().last().unwrap_or(1) as usize,
    );
    for (x, y) in predictions.iter_cols().zip(targets.iter_cols()) {
        let loss: f64 = x
            .iter()
            .cloned()
            .zip(y.iter().cloned())
            .map(|(mut x, y)| -> f64 {
                // clamp x to prevent division by 0
                // and prevent the dragging of the mean error later
                const MAX: f64 = 1.0 - 1e-7;
                const MIN: f64 = 1e-7;
                x = if x < MAX { x } else { MAX };
                x = if x > MIN { x } else { MIN };
                x.log(E) * y
            })
            .sum();
        // loss is always < 0 so let's flip it...
        out.push(-loss);
    }

    let res = NdArray::new_with_values(out.len() as u32, out.into_boxed_slice())?;
    Ok(res)
}
