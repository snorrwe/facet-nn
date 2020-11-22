import nd


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

        self.weights = nd.array([[0] * out] * inp)
        self.biases = nd.array([0] * out)
        self.activation = activation

    def forward(self, inp):
        return self.activation(inp.matmul(self.weights) + self.biases)


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


class Loss:
    def __init__(self, lossfn):
        assert callable(lossfn)
        self.loss = lossfn

    def calculate(self, pred, target):
        assert pred.shape == target.shape

        losses = self.loss(pred, target)
        data_loss = losses.mean()
        return data_loss


def accuracy(pred, target):
    pred = pred.argmax()
    target = target.argmax()
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
