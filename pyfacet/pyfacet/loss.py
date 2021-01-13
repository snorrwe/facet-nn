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

    def calculate(self, pred, target):
        losses = self.forward(pred, target)
        return pf.mean(list(losses))[0], self.regularization_loss()

    def regularization_loss(self):
        r_loss = 0
        for l in self.trainable_layers:
            if l.weight_regularizer_l1 is not None and l.weight_regularizer_l1 > 0:
                r_loss += layer.weight_regularizer_l1 * pf.sum(pf.abs(l.weights))
            if l.weight_regularizer_l2 is not None and l.weight_regularizer_l2 > 0:
                r_loss += layer.weight_regularizer_l2 * pf.sum(l.weights * l.weights)
            if l.bias_regularizer_l1 is not None and l.bias_regularizer_l1 > 0:
                r_loss += layer.bias_regularizer_l1 * pf.sum(pf.abs(l.biases))
            if l.bias_regularizer_l2 is not None and l.bias_regularizer_l2 > 0:
                r_loss += layer.bias_regularizer_l2 * pf.sum(l.biases * l.biases)
        return r_loss

    def forward(self, pred, target):
        assert pred.shape == target.shape
        sample_losses = self.loss(pred, target)
        self.output = sample_losses
        return sample_losses

    def backward(self, dvalues, target):
        self.dinputs = self.dlossfn(dvalues, target)


class BinaryCrossentropy(Loss):
    def forward(self, pred, target):
        pred_clipped = pf.clip(pred, 1e-7, 1 - 1e-7)

        tp = target * pf.log(pred_clipped)
        t1 = scalar(1) - target
        t2 = pf.log(scalar(1) - pred_clipped)

        res = scalar(-1) * (tp + t1 * t2)
        self.output = res
        return res

    def backward(self, dvalues, target):

        samples, outputs = dvalues.shape[:2]

        # clip data to prevent division by zero
        clipped_dvalues = pf.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = (
            scalar(-1)
            * (
                target / clipped_dvalues
                - (scalar(1) - target) / (scalar(1) - clipped_dvalues)
            )
            / scalar(outputs)
        )

        self.dinputs = self.dinputs / scalar(samples)


class MeanSquaredError(Loss):
    def forward(self, pred, target):
        sample_losses = pf.mean((target - pred) ** 2)
        # flatten
        sample_losses = sample_losses.reshape([pred.shape[0]])
        self.output = sample_losses
        return self.output

    def backward(self, dvalues, target):
        samples, outputs = dvalues.shape

        self.dinputs = scalar(-2) * (target - dvalues) / scalar(outputs)
        self.dinputs /= scalar(samples)
