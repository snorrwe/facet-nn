from nnfs import *
import pydu


layer1 = DenseLayer(3, 3, activation=pydu.relu, dactifn=pydu.drelu_dz)

inp = pydu.array([[1, 1, 1]] * 9)

out = layer1.forward(inp)

print("out", repr(out))

layer1.backward(out)

res = cce_backward(out, inp)

print("back", res)

# softmax test

softmax_out = pydu.array([0.7, 0.1, 0.2])
