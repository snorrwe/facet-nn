from . import pyfacet as pf


class Accuracy:
    def calculate(self, pred, y):
        comparisions = self.compare(pred, y)
        # global mean is a scalar
        # flatten the output and return the inner scalar value
        return pf.mean(list(comparisions.as_f64()))[0]


class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, y, force=False):
        if self.precision is None or force:
            self.precision = pf.std(y) / pf.scalar(250.0)

    def compare(self, pred, y):
        return pf.abs(pred - y) < self.precision


class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, force=False):
        pass

    def compare(self, pred, y):
        if not self.binary and len(y.shape) == 2:
            y = pf.argmax(y)
            pred = pf.argmax(pred)
        return pred == y
