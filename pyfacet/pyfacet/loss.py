from . import pyfacet as pf


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
