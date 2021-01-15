from .pyfacet import binomial, scalar
from .pyfacet import DenseLayer  # reexport


class InputLayer:
    def forward(self, inputs):
        self.output = inputs


class DropoutLayer:
    def __init__(self, rate):
        self.rate = 1 - rate
        self.training_only = True

    def forward(self, inputs):
        self.inputs = inputs
        self.mask = binomial(1, self.rate, inputs.shape) / scalar(self.rate)
        self.output = inputs * self.mask
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask
