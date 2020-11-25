from operator import mul
import pydu


class DenseLayer:
    """
    Dense layers are interconnected, all inputs are connected to all outputs
    """

    def __init__(self, inp, out):
        """
        :param inp: number of input neurons
        :param out: number of output neurons
        """

        self.weights = pydu.array([[69] * out] * inp)
        self.biases = pydu.array([42] * out)
        self.last_input = None

    def forward(self, inp):
        self.last_input = inp
        return inp.matmul(self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = self.last_input.transpose().matmul(dvalues)
        self.dinputs = dvalues.matmul(self.weights.transpose())
        self.dbiases = pydu.sum(dvalues)

    def __repr__(self):
        return f"DenseLayer weights: {self.weights.shape} biases: {self.biases.shape}"


class Network:
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations

        assert len(layers) == len(activations)

    def forward(self, x):
        for (layer, activation) in zip(self.layers, self.activations):
            x = layer.forward(x)
            x = activation(x)
        return x

    def __repr__(self):
        layers = "\n".join((repr(x) for x in self.layers))
        return f"Network object of {len(self.layers)} Layers:\n{layers}"


class Activation:
    def __init__(self, fn):
        assert callable(fn)
        self.fn = fn

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.fn(inputs)
        return self.output


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
        self.dinputs = dvalues.clone()
        self.dinputs = self.dinputs - target
        self.dinputs = self.dinputs / pydu.array([samples]).reshape([0])


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
