from nnfs import *
import nd


layer1 = DenseLayer(3, 3, activation=nd.softmax,)

inp = nd.array([[1, 1, 1]] * 9)

out = layer1.forward(inp)

print(repr(out))
