from . import pyfacet as pf
from .pyfacet import scalar


class Loss:
    def __init__(self, lossfn, dlossfn=None):
        assert callable(lossfn)
        if dlossfn:
            assert callable(dlossfn)
        self.loss = lossfn
        self.dloss = dlossfn

    def calculate(self, pred, target):
        assert pred.shape == target.shape

        losses = self.loss(pred, target)
        return pf.mean(losses)

    def backward(self, dvalues, target):
        self.dinputs = self.dlossfn(dvalues, target)


class BinaryCrossentropy:
    def forward(self, pred, target):
        pred_clipped = pf.clip(pred, 1e-7, 1 - 1e-7)

        sample_losses = scalar(-1) * (
            target * pf.log(pred_clipped)
            + (scalar(1) - target) * pf.log(scalar(1) - pred_clipped)
        )

        sample_losses = pf.mean(sample_losses)
        return sample_losses
