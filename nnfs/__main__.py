from nnfs import *
import nd


layer1 = DenseLayer(
    weights=nd.array(list(range(9))).reshape([3, 3]),
    biases=nd.array(list(range(4, 7))),
    activation=nd.softmax,
)

inp = nd.array([[1, 1, 1]] * 9)

out = layer1.forward(inp)

print(repr(out))
