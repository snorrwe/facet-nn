class ActivationLinear:
    """
    Linear activation function

    ```
    y = x
    ```
    """

    def forward(self, inp):
        self.inputs = inp
        self.output = inp

    def backward(self, dvalues):
        self.dinputs = dvalues.clone()


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


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation(pyfacet.softmax)
        self.loss = Loss(pyfacet.categorical_cross_entropy)

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
        s = pyfacet.scalar(samples)
        self.dinputs = self.dinputs / s
