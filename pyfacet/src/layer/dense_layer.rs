use crate::pyndarray::NdArrayD;
use facet_core::layer::dense_layer::DenseLayer as CoreLayer;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
#[derive(Clone)]
pub struct DenseLayer {
    inner: facet_core::layer::dense_layer::DenseLayer,
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
        Ok(Self {
            inner: CoreLayer::new(inputs, outputs).with_training(
                weight_regularizer_l1,
                weight_regularizer_l2,
                bias_regularizer_l1,
                bias_regularizer_l2,
            ),
            id: uuid::Uuid::new_v4(),
        })
    }
    #[getter]
    pub fn weight_regularizer_l1(&self) -> Option<f64> {
        self.inner
            .training
            .as_ref()
            .and_then(|t| t.weight_regularizer_l1)
    }
    #[getter]
    pub fn weight_regularizer_l2(&self) -> Option<f64> {
        self.inner
            .training
            .as_ref()
            .and_then(|t| t.weight_regularizer_l2)
    }
    #[getter]
    pub fn bias_regularizer_l1(&self) -> Option<f64> {
        self.inner
            .training
            .as_ref()
            .and_then(|t| t.bias_regularizer_l1)
    }
    #[getter]
    pub fn bias_regularizer_l2(&self) -> Option<f64> {
        self.inner
            .training
            .as_ref()
            .and_then(|t| t.bias_regularizer_l2)
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn biases(&self) -> NdArrayD {
        NdArrayD {
            inner: self.inner.biases.clone(),
        }
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn weights(&self) -> NdArrayD {
        NdArrayD {
            inner: self.inner.weights.clone(),
        }
    }

    #[setter]
    pub fn set_biases(&mut self, b: NdArrayD) {
        self.inner.biases = b.inner;
    }

    #[setter]
    pub fn set_weights(&mut self, w: NdArrayD) {
        self.inner.weights = w.inner;
    }

    #[getter]
    pub fn id(&self) -> String {
        self.id.to_string()
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn output(&self) -> NdArrayD {
        NdArrayD {
            inner: self.inner.output.clone(),
        }
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn dweights(&self) -> Option<NdArrayD> {
        self.inner
            .training
            .as_ref()
            .map(|t| &t.dweights)
            .map(|o| NdArrayD { inner: o.clone() })
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn dbiases(&self) -> Option<NdArrayD> {
        self.inner
            .training
            .as_ref()
            .map(|t| &t.dbiases)
            .map(|o| NdArrayD { inner: o.clone() })
    }

    /// Copies the output.
    ///
    /// TODO: return view
    #[getter]
    pub fn dinputs(&self) -> Option<NdArrayD> {
        self.inner
            .training
            .as_ref()
            .map(|t| &t.dinputs)
            .map(|o| NdArrayD { inner: o.clone() })
    }

    pub fn forward(&mut self, inputs: NdArrayD) -> PyResult<()> {
        let inputs = inputs.inner;
        self.inner
            .forward(inputs)
            .map_err(|err| PyValueError::new_err(format!("Failed to forward {}", err)))
    }

    /// Consumes the last `inputs` replacing it with `None`.
    pub fn backward(&mut self, dvalues: NdArrayD) -> PyResult<()> {
        let dvalues = dvalues.inner;
        self.inner
            .backward(dvalues)
            .map_err(|err| PyValueError::new_err(format!("Failed to back propagate {}", err)))
    }
}
