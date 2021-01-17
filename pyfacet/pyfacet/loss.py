from . import pyfacet as pf
from .pyfacet import scalar


class Loss:
    def __init__(self, lossfn=None, dlossfn=None):
        if lossfn:
            assert callable(lossfn)
        if dlossfn:
            assert callable(dlossfn)
        self.loss = lossfn
        self.dloss = dlossfn
        self.trainable_layers = []

    def calculate(self, pred, y, *, include_regularization=False):
        losses = self.forward(pred, y)
        data_loss = pf.mean(list(losses))[0]
        if not include_regularization:
            return data_loss, None
        return data_loss, self.regularization_loss()

    def regularization_loss(self):
        r_loss = 0
        for l in self.trainable_layers:
            if l.weight_regularizer_l1 is not None and l.weight_regularizer_l1 > 0:
                r_loss += l.weight_regularizer_l1 * pf.sum(pf.abs(l.weights))[0]
            if l.weight_regularizer_l2 is not None and l.weight_regularizer_l2 > 0:
                r_loss += l.weight_regularizer_l2 * pf.sum(l.weights * l.weights)[0]
            if l.bias_regularizer_l1 is not None and l.bias_regularizer_l1 > 0:
                r_loss += l.bias_regularizer_l1 * pf.sum(pf.abs(l.biases))[0]
            if l.bias_regularizer_l2 is not None and l.bias_regularizer_l2 > 0:
                r_loss += l.bias_regularizer_l2 * pf.sum(l.biases * l.biases)[0]
        return r_loss

    def forward(self, pred, y):
        assert pred.shape == y.shape
        sample_losses = self.loss(pred, y)
        self.output = sample_losses
        return sample_losses

    def backward(self, dvalues, y):
        self.dinputs = self.dloss(dvalues, y)
        return self.dinputs


class BinaryCrossentropy(Loss):
    def forward(self, pred, y):
        pred_clipped = pf.clip(pred, 1e-7, 1 - 1e-7)

        tp = y * pf.log(pred_clipped)
        t1 = scalar(1) - y
        t2 = pf.log(scalar(1) - pred_clipped)

        res = scalar(-1) * (tp + t1 * t2)
        self.output = res
        return res

    def backward(self, dvalues, y):

        samples, outputs = dvalues.shape[:2]

        # clip data to prevent division by zero
        clipped_dvalues = pf.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = (
            scalar(-1)
            * (y / clipped_dvalues - (scalar(1) - y) / (scalar(1) - clipped_dvalues))
            / scalar(outputs)
        )

        self.dinputs = self.dinputs / scalar(samples)


class MeanSquaredError(Loss):
    def forward(self, pred, y):
        sample_losses = pf.mean((y - pred) ** 2)
        # flatten
        sample_losses = sample_losses.reshape([pred.shape[0]])
        self.output = sample_losses
        return self.output

    def backward(self, dvalues, y):
        samples, outputs = dvalues.shape

        self.dinputs = scalar(-2) * (y - dvalues) / scalar(outputs)
        self.dinputs /= scalar(samples)
