from nnfs import *
import nd


layer1 = DenseLayer(
    weights=nd.NdArrayD([3, 3], list(range(9))),
    biases=nd.NdArrayD([3], list(range(4, 7))),
    activation=nd.softmax,
)


inp = nd.NdArrayD([9, 3], [1, 1, 1] * 9)


out = layer1.forward(inp)

print(repr(out))
