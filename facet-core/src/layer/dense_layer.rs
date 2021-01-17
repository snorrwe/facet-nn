use crate::ndarray::{NdArray, NdArrayError};
use rand::Rng;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Clone)]
pub struct DenseLayer {
    // core attributes
    pub weights: NdArray<f32>,
    pub biases: NdArray<f32>,
    pub output: NdArray<f32>,

    pub training: Option<Box<DenseLayerTraining>>,
}

/// Holds data related to back propagation / training
#[derive(Clone, Default)]
pub struct DenseLayerTraining {
    // memoization for training purposes
    pub inputs: NdArray<f32>,
    // training data
    pub dweights: NdArray<f32>,
    pub dbiases: NdArray<f32>,
    pub dinputs: NdArray<f32>,
    // hyperparameters
    pub weight_regularizer_l1: Option<f32>,
    pub weight_regularizer_l2: Option<f32>,
    pub bias_regularizer_l1: Option<f32>,
    pub bias_regularizer_l2: Option<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum DenseLayerError {
    #[error("Failed to perform matrix multiplication {0}")]
    MatMulFail(NdArrayError),
    #[error("No inputs available. Perhaps you forgot to call `forward`?")]
    NoInputs,
}

impl DenseLayer {
    pub fn new(inputs: u32, outputs: u32) -> Self {
        let weights = NdArray::new_with_values(
            [inputs, outputs],
            (0..inputs as usize * outputs as usize)
                .map(|_| rand::thread_rng().gen_range(-1., 1.))
                .collect(),
        )
        .unwrap();

        let biases = NdArray::new_with_values(
            outputs,
            (0..outputs as usize)
                .map(|_| rand::thread_rng().gen_range(-1., 1.))
                .collect(),
        )
        .unwrap();

        Self {
            weights,
            biases,
            output: Default::default(),
            training: None,
        }
    }

    pub fn with_training(
        mut self,
        weight_regularizer_l1: Option<f32>,
        weight_regularizer_l2: Option<f32>,
        bias_regularizer_l1: Option<f32>,
        bias_regularizer_l2: Option<f32>,
    ) -> Self {
        self.training = Some(Box::new(DenseLayerTraining {
            weight_regularizer_l1,
            weight_regularizer_l2,
            bias_regularizer_l1,
            bias_regularizer_l2,
            ..Default::default()
        }));
        self
    }

    pub fn forward(&mut self, inputs: NdArray<f32>) -> Result<(), DenseLayerError> {
        assert!(
            matches!(inputs.shape(), crate::prelude::Shape::Matrix(_)),
            "Forward input must be a matrix"
        );
        self.output.reshape(0);

        inputs
            .matmul_f32(&self.weights, &mut self.output)
            .map_err(DenseLayerError::MatMulFail)?;

        assert_eq!(self.output.shape().last(), self.biases.shape().last());

        let biases = self.biases.as_slice();
        self.output.iter_rows_mut().for_each(|col| {
            col.iter_mut()
                .zip(biases.iter())
                .for_each(|(out, bias)| *out += bias)
        });

        if let Some(ref mut t) = self.training {
            t.inputs = inputs;
        }

        Ok(())
    }

    /// Consumes the last `inputs` replacing it with an empty array.
    pub fn backward(&mut self, dvalues: NdArray<f32>) -> Result<(), DenseLayerError> {
        let inputs = self
            .training
            .as_mut()
            .map(|t| std::mem::take(&mut t.inputs))
            .ok_or(DenseLayerError::NoInputs)?;

        // we know that training is some at this point
        let training = self.training.as_mut().unwrap();

        inputs
            .transpose()
            .matmul_f32(&dvalues, &mut training.dweights)
            .map_err(DenseLayerError::MatMulFail)?;

        let s = dvalues.clone().transpose();
        training.dbiases = crate::sum(&s);

        // Regularization
        if let Some(l1) = training.weight_regularizer_l1 {
            regularize_l1(l1, &mut training.dweights, &self.weights);
        }
        if let Some(l2) = training.weight_regularizer_l2 {
            regularize_l2(l2, &mut training.dweights, &self.weights);
        }

        if let Some(l1) = training.bias_regularizer_l1 {
            regularize_l1(l1, &mut training.dbiases, &self.biases);
        }
        if let Some(l2) = training.bias_regularizer_l2 {
            regularize_l2(l2, &mut training.dbiases, &self.biases);
        }

        // Gradients
        dvalues
            .matmul_f32(&self.weights.clone().transpose(), &mut training.dinputs)
            .map_err(DenseLayerError::MatMulFail)?;

        Ok(())
    }
}

fn regularize_l1(l1: f32, to_regulate: &mut NdArray<f32>, inp: &NdArray<f32>) {
    let mut d_l1 = NdArray::new_with_values(
        inp.shape().clone(),
        smallvec::smallvec![
            1.0; inp.shape().span()
        ],
    )
    .unwrap();

    d_l1.as_mut_slice()
        .iter_mut()
        .enumerate()
        .filter(|(i, _)| inp.as_slice()[*i] < 0.0)
        .for_each(|(_, x)| *x = -1.0);

    #[cfg(feature = "rayon")]
    {
        d_l1.as_mut_slice().par_iter_mut().for_each(|x| *x *= l1);
    }
    #[cfg(not(feature = "rayon"))]
    {
        d_l1.as_mut_slice().iter_mut().for_each(|x| *x *= l1);
    }

    *to_regulate = to_regulate.add(&d_l1).unwrap();
}

fn regularize_l2(l2: f32, to_regulate: &mut NdArray<f32>, inp: &NdArray<f32>) {
    let mul = inp.mul(&NdArray::new_scalar(l2)).unwrap();
    *to_regulate = to_regulate.add(&mul).unwrap();
}
