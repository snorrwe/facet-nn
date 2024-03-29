use crate::{
    ndarray::NdArray,
    ndarray::{matrix::matmul_impl, shape::Shape},
    DuResult,
};
use std::f32::consts::E;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "rayon")]
pub fn relu(inp: &NdArray<f32>) -> NdArray<f32> {
    let mut out = inp.clone();
    out.par_iter_rows_mut().for_each(|row| {
        for v in row {
            *v = v.max(0.0);
        }
    });
    out
}

#[cfg(not(feature = "rayon"))]
pub fn relu(inp: &NdArray<f32>) -> NdArray<f32> {
    inp.map(|v| v.max(0.0))
}

/// ReLU derivative
#[cfg(feature = "rayon")]
pub fn drelu_dz(inputs: &NdArray<f32>, dvalues: &NdArray<f32>) -> NdArray<f32> {
    let mut res = dvalues.clone();
    res.iter_rows_mut()
        .zip(inputs.iter_rows())
        .par_bridge()
        .for_each(|(dx, dz)| _drelu(dx, dz));
    res
}

fn _drelu(dx: &mut [f32], dz: &[f32]) {
    debug_assert_eq!(dx.len(), dz.len());
    for i in 0..dx.len() {
        dx[i] *= (dz[i] > 0.0) as i32 as f32;
    }
}

/// ReLU derivative
#[cfg(not(feature = "rayon"))]
pub fn drelu_dz(inputs: &NdArray<f32>, dvalues: &NdArray<f32>) -> NdArray<f32> {
    // #[cfg(feature = "rayon")]
    let mut res = dvalues.clone();
    res.iter_rows_mut()
        .zip(inputs.iter_rows())
        .for_each(|(dx, dz)| _drelu(dx, dz));

    res
}

/// Inp is interpreted as a either a collection of vectors, applying softmax to each row or as a
/// single vector.
///
/// Scalars will always return 1
pub fn softmax(inp: &NdArray<f32>) -> DuResult<NdArray<f32>> {
    // softmax of a scalar value is always 1.
    if matches!(inp.shape(), Shape::Scalar(_)) {
        return Ok(NdArray::new_with_values(0, [1.0][..].into())?);
    }
    // else treat the input as a collection of vectors
    let mut it = inp.as_slice().iter().cloned();
    let first = it.next().expect("no value");
    let max: f32 = it.fold(first, |max, value| value.max(max));

    let expvalues = inp
        .sub(&NdArray::from(max))
        .expect("Failed to sub max from the input")
        .map(|v: &f32| {
            let res = E.powf(*v);
            if res.is_nan() || res == 0.0 {
                // very large V's will produce an output of 0 which will be bad down the line
                1e-12
            } else {
                res
            }
        });

    let mut norm_base: NdArray<f32> = expvalues
        .iter_rows()
        .map(|row| row.iter().cloned().sum())
        .collect();

    norm_base.reshape([norm_base.shape().span() as u32, 1]);

    let mut res = expvalues;
    for (norm, row) in norm_base.iter_rows().zip(res.iter_rows_mut()) {
        debug_assert_eq!(norm.len(), 1);
        row.iter_mut().for_each(|v| *v /= norm[0]);
    }
    Ok(res)
}

/// Softmax backwards pass, calculating gradient
///
///
#[cfg(feature = "rayon")]
pub fn dsoftmax(output: &NdArray<f32>, dvalues: &NdArray<f32>) -> DuResult<NdArray<f32>> {
    let mut res = NdArray::new(dvalues.shape().clone());

    let collen = output.shape().last();

    res.as_mut_slice()
        .par_chunks_mut(dvalues.shape().last() as usize)
        .enumerate()
        .try_for_each(|(i, row)| {
            let output = output.get_row(&[i as u32]).unwrap();
            let dvalues = dvalues.get_row(&[i as u32]).unwrap();

            let mut jacobian_matrix = NdArray::new([collen, collen]);
            let mut dotcache = NdArray::new([collen, collen]);
            diagflat(output, &mut jacobian_matrix);
            matmul_impl([collen, 1, collen], output, output, dotcache.as_mut_slice())?;

            jacobian_matrix = jacobian_matrix.sub(&dotcache)?;

            matmul_impl(
                [collen, collen, 1],
                jacobian_matrix.as_slice(),
                dvalues,
                row,
            )
        })?;

    Ok(res)
}

fn diagflat(output: &[f32], mat: &mut NdArray<f32>) {
    for i in 0..output.len() {
        for j in 0..output.len() {
            let i = i as u32;
            let j = j as u32;
            // branchless setter, if the item is not in the diagonal set it to 0
            *mat.get_mut(&[i, j]).unwrap() = output[i as usize] * (i == j) as u32 as f32;
        }
    }
}

pub fn sigmoid(input: &NdArray<f32>, output: &mut NdArray<f32>) -> DuResult<()> {
    output.reshape(input.shape().clone());
    for (x, y) in input
        .as_slice()
        .iter()
        .zip(output.as_mut_slice().iter_mut())
    {
        *y = 1.0 / (1. + E.powf(-x))
    }
    Ok(())
}

/// Derivative of sigmoid function
///
/// dvalues * (1-output) * output
pub fn dsigmoid(output: &NdArray<f32>, dvalues: &NdArray<f32>) -> DuResult<NdArray<f32>> {
    let v = dvalues.mul(&(NdArray::new_scalar(1.).sub(output)?))?;
    Ok(v.mul(output)?)
}
