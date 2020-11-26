from .optimizers import *
from operator import mul
import pydu


class DenseLayer:
    """
    Dense layers are interconnected, all inputs are connected to all outputs
    """

    def __init__(self, inp, out, name=None):
        """
        :param inp: number of input neurons
        :param out: number of output neurons
        """

        self.weights = pydu.array([[0.69] * out] * inp)
        self.biases = pydu.array([0.42] * out)
        self.inputs = None
        self.name = name

    def forward(self, inp):
        self.inputs = inp
        z = inp.matmul(self.weights)
        self.output = z + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = self.inputs.T.matmul(dvalues)
        self.dbiases = pydu.sum(dvalues.T)
        self.dinputs = dvalues.matmul(self.weights.T)

    def __repr__(self):
        return (
            f"DenseLayer {self.name}\nweights:\n{self.weights}\nbiases:\n{self.biases}"
        )

    def __str__(self):
        return f"DenseLayer {self.name} weights: {self.weights.shape} biases: {self.biases.shape}"


class Activation:
    def __init__(self, fn, df=None):
        assert callable(fn)
        if df:
            assert callable(df)
        self.fn = fn
        self.df = df

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.fn(inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = self.df(self.inputs, dvalues)


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
        return losses.mean()

    def backward(self, dvalues, target):
        self.dinputs = self.dlossfn(dvalues, target)


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation(pydu.softmax)
        self.loss = Loss(pydu.categorical_cross_entropy)

    def forward(self, inputs, target):
        y = self.activation.forward(inputs)
        self.output = y
        return self.loss.calculate(y, target)

    def backward(self, dvalues, target):
        """
        optimized backwards step

        :param target: at this time we assume that target it 1-hot
        """
        samples = dvalues.shape[0]
        self.dinputs = dvalues - target
        s = pydu.scalar(samples)
        self.dinputs = self.dinputs / s


def labels_to_y(labels):
    """
    convert a list of labels to a dense target matrix
    """
    lset = set(labels)
    label_pos = {label: i for (i, label) in enumerate(lset)}
    n_classes = len(lset)

    res = []
    for label in labels:
        v = [0.0] * n_classes
        v[label_pos[label]] = 1.0
        res.append(v)

    return pydu.array(res)


def accuracy(pred, target):
    """
    How often the prediction matches the target, on average
    """
    # collapse dense data into sparse data
    if len(target.shape) == 2:
        target = target.argmax()
    pred = pred.argmax()
    diff = pred == target
    return diff.as_f64().mean()


def first_n(n, it):
    """
    :param it: iterable
    :return: the first n items in it
    """

    try:
        for _ in range(n):
            yield next(it)
    except StopIteration:
        return


def reduce(op, arr, init):
    for x in arr:
        init = op(init, x)
    return init


def cce_backward(z, y):
    samples = reduce(mul, z.shape[:-1], 1)

    gradient = (y / z) * pydu.array([-1.0]).reshape([0])
    # Normalize gradient
    return gradient / pydu.array([samples]).reshape([0])
