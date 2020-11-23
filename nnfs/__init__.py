import pydu


class DenseLayer:
    """
    Dense layers are interconnected, all inputs are connected to all outputs
    """

    def __init__(self, inp, out, activation):
        """
        :param inp: number of input neurons
        :param out: number of output neurons
        """
        assert callable(activation)

        self.weights = pydu.array([[0] * out] * inp)
        self.biases = pydu.array([0] * out)
        self.activation = activation

    def forward(self, inp):
        return self.activation(inp.matmul(self.weights) + self.biases)

    def __repr__(self):
        return f"DenseLayer weights: {self.weights.shape} biases: {self.biases.shape} activation: {self.activation}"


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __repr__(self):
        layers = "\n".join((repr(x) for x in self.layers))
        return f"Network object of {len(self.layers)} Layers:\n{layers}"


class Loss:
    def __init__(self, lossfn):
        assert callable(lossfn)
        self.loss = lossfn

    def calculate(self, pred, target):
        assert pred.shape == target.shape

        losses = self.loss(pred, target)
        return losses.mean()


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
