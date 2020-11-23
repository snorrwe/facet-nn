from nnfs import *
import pydu


layer1 = DenseLayer(3, 3, activation=pydu.softmax)

inp = pydu.array([[1, 1, 1]] * 9)

out = layer1.forward(inp)

print(repr(out))
print(repr(out.transpose()))
