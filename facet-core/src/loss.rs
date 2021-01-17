use std::f32::consts::E;

use crate::{ndarray::Data, ndarray::NdArray, DuError, DuResult};

/// The author recommends running `softmax` on the output before calling this function
///
/// Expect the predictions to be in the interval [0, 1]
pub fn categorical_cross_entropy(
    predictions: &NdArray<f32>,
    targets: &NdArray<f32>,
) -> DuResult<NdArray<f32>> {
    if predictions.shape() != targets.shape() {
        return Err(DuError::MismatchedShapes(
            predictions.shape().clone(),
            targets.shape().clone(),
        ));
    }

    let mut out = Data::with_capacity(
        predictions.shape().span() / predictions.shape().last().max(1) as usize,
    );
    for (x, y) in predictions.iter_rows().zip(targets.iter_rows()) {
        let loss: f32 = x
            .iter()
            .cloned()
            .zip(y.iter().cloned())
            .map(|(mut x, y)| -> f32 {
                // clamp x to prevent division by 0
                // and prevent the dragging of the mean error later
                const MAX: f32 = 1.0 - 1e-7;
                const MIN: f32 = 1e-7;
                x = if x < MAX { x } else { MAX };
                x = if x > MIN { x } else { MIN };
                x.log(E) * y
            })
            .sum();
        // loss is always < 0 so let's flip it...
        out.push(-loss);
    }

    let res = NdArray::new_with_values(out.len() as u32, out)?;
    Ok(res)
}
