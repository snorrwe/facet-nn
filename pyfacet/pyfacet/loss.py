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

        tp = target * pf.log(pred_clipped)
        t1 = scalar(1) - target
        t2 = pf.log(scalar(1) - pred_clipped)

        print(
            "??????????????????",
            tp,
            t1,
            t2,
            pred_clipped,
            scalar(1) - pred_clipped,
            sep="\n",
        )

        sample_losses = scalar(-1) * (tp + t1 * t2)

        sample_losses = pf.mean(sample_losses)
        return sample_losses

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

        print("1", self.dinputs)
        self.dinputs = self.dinputs / scalar(samples)

        print(dvalues, dvalues.shape, samples, outputs)
        print("2", self.dinputs)
