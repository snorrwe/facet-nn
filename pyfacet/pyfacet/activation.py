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
