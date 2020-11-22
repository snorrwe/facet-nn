import nd


class DenseLayer:
    def __init__(self, weights, biases, activation):
        self.weights = weights
        self.biases = biases
        self.activation = activation

        assert [
            weights.shape[-1]
        ] == biases.shape, (
            "Expected weights and biases to accept the same input dimensions"
        )

        assert callable(activation)

    def forward(self, inp):
        return self.activation(inp.matmul(self.weights) + self.biases)
