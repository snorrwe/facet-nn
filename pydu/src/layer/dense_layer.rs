use crate::pyndarray::NdArrayD;
use du_core::ndarray::NdArray;
use pyo3::{exceptions::PyValueError, prelude::*};

use du_core::rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct DenseLayer {
    // core attributes
    weights: NdArray<f64>,
    biases: NdArray<f64>,
    // memoization for training purpuses
    inputs: Option<NdArray<f64>>,
    output: Option<NdArray<f64>>,
    // training data
    dweights: Option<NdArray<f64>>,
    dbiases: Option<NdArray<f64>>,
    dinputs: Option<NdArray<f64>>,
    // hyperparameters
    weight_regularizer_l1: Option<f64>,
    weight_regularizer_l2: Option<f64>,
    bias_regularizer_l1: Option<f64>,
    bias_regularizer_l2: Option<f64>,

    id: uuid::Uuid,
}

#[pymethods]
impl DenseLayer {
    #[new(
        "weight_regularizer_l1=None",
        "weight_regularizer_l2=None",
        "bias_regularizer_l1=None",
        "bias_regularizer_l2=None"
    )]
    pub fn new(
        inputs: u32,
        outputs: u32,
        weight_regularizer_l1: Option<f64>,
        weight_regularizer_l2: Option<f64>,
        bias_regularizer_l1: Option<f64>,
        bias_regularizer_l2: Option<f64>,
    ) -> PyResult<Self> {
        let weights = NdArray::new_with_values(
            [inputs, outputs],
            du_core::smallvec::smallvec![ 0.69; inputs as usize * outputs as usize ],
        )
        .unwrap();

        let biases = NdArray::new_with_values(
            outputs,
            du_core::smallvec::smallvec![ 0.42;outputs as usize ],
        )
        .unwrap();
        Ok(Self {
            weights,
            biases,

            inputs: None,
            output: None,
            dweights: None,
            dbiases: None,
            dinputs: None,

            weight_regularizer_l1,
            weight_regularizer_l2,
            bias_regularizer_l1,
            bias_regularizer_l2,

            id: uuid::Uuid::new_v4(),
        })
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn biases(&self) -> NdArrayD {
        NdArrayD {
            inner: self.biases.clone(),
        }
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn weights(&self) -> NdArrayD {
        NdArrayD {
            inner: self.weights.clone(),
        }
    }

    #[setter]
    pub fn set_biases(&mut self, b: NdArrayD) {
        self.biases = b.inner;
    }

    #[setter]
    pub fn set_weights(&mut self, w: NdArrayD) {
        self.weights = w.inner;
    }

    #[getter]
    pub fn id(&self) -> String {
        self.id.to_string()
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn output(&self) -> Option<NdArrayD> {
        self.output.as_ref().map(|o| NdArrayD { inner: o.clone() })
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn dweights(&self) -> Option<NdArrayD> {
        self.dweights
            .as_ref()
            .map(|o| NdArrayD { inner: o.clone() })
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn dbiases(&self) -> Option<NdArrayD> {
        self.dbiases.as_ref().map(|o| NdArrayD { inner: o.clone() })
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn dinputs(&self) -> Option<NdArrayD> {
        self.dinputs.as_ref().map(|o| NdArrayD { inner: o.clone() })
    }

    pub fn forward(&mut self, inputs: NdArrayD) -> PyResult<()> {
        let inputs = inputs.inner;

        if self.output.is_none() {
            self.output = Some(NdArray::new(0));
        }

        let biases = self.biases.as_slice();
        let output = self.output.as_mut().unwrap();

        inputs.matmul(&self.weights, output).map_err(|err| {
            PyValueError::new_err(format!("Failed to multiply weights with the input {}", err))
        })?;

        assert_eq!(output.shape().last(), self.biases.shape().last());

        output.iter_cols_mut().for_each(|col| {
            col.iter_mut()
                .zip(biases.iter())
                .for_each(|(out, bias)| *out += bias)
        });

        self.inputs = Some(inputs);

        Ok(())
    }

    /// Consumes the last `inputs` replacing it with `None`.
    pub fn backward(&mut self, py: Python, dvalues: NdArrayD) -> PyResult<()> {
        let inputs = self.inputs.take().ok_or_else(|| {
            PyValueError::new_err(format!(
                "No inputs available. Perhaps you forgot to call `forward`?"
            ))
        })?;

        if self.dweights.is_none() {
            self.dweights = Some(NdArray::new(0));
        }
        inputs
            .transpose()
            .matmul(&dvalues.inner, self.dweights.as_mut().unwrap())
            .map_err(|err| {
                PyValueError::new_err(format!("Failed to multiply inputs T with dvalues {}", err))
            })?;

        let s = dvalues.transpose()?;
        let s = Py::new(py, s)?;
        // Cast as Any
        let s = unsafe { Py::from_owned_ptr(py, s.into_ptr()) };
        self.dbiases = Some(crate::sum(py, s)?.inner);

        // Regularization
        if let Some(l1) = self.weight_regularizer_l1 {
            regularize_l1(l1, self.dweights.as_mut().unwrap(), &self.weights)?;
        }
        if let Some(l2) = self.weight_regularizer_l2 {
            regularize_l2(l2, self.dweights.as_mut().unwrap(), &self.weights)?;
        }

        if let Some(l1) = self.bias_regularizer_l1 {
            regularize_l1(l1, self.dbiases.as_mut().unwrap(), &self.biases)?;
        }
        if let Some(l2) = self.bias_regularizer_l2 {
            regularize_l2(l2, self.dbiases.as_mut().unwrap(), &self.biases)?;
        }

        // Gradients
        if self.dinputs.is_none() {
            self.dinputs = Some(NdArray::new(0));
        }
        dvalues
            .inner
            .matmul(
                &self.weights.clone().transpose(),
                self.dinputs.as_mut().unwrap(),
            )
            .map_err(|err| {
                PyValueError::new_err(format!("Failed to multiply dvalues with weights T {}", err))
            })?;

        Ok(())
    }
}

fn regularize_l1(l1: f64, to_regulate: &mut NdArray<f64>, inp: &NdArray<f64>) -> PyResult<()> {
    let mut d_l1 = NdArray::new_with_values(
        inp.shape().clone(),
        du_core::smallvec::smallvec![
            1.0; inp.shape().span()
        ],
    )
    .unwrap();

    d_l1.as_mut_slice()
        .iter_mut()
        .enumerate()
        .filter(|(i, _)| inp.as_slice()[*i] < 0.0)
        .for_each(|(_, x)| *x = -1.0);

    d_l1.as_mut_slice().par_iter_mut().for_each(|x| *x *= l1);

    *to_regulate = to_regulate.add(&d_l1).unwrap();
    Ok(())
}

fn regularize_l2(l2: f64, to_regulate: &mut NdArray<f64>, inp: &NdArray<f64>) -> PyResult<()> {
    let mul = inp.mul(&NdArray::new_scalar(l2)).unwrap();
    *to_regulate = to_regulate.add(&mul).unwrap();
    Ok(())
}
