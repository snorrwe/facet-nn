from nnfs import *
import pydu


boi = Activation_Softmax_Loss_CategoricalCrossentropy()


x = pydu.array([[1, 2, 8] * 8])
y = pydu.array([[1, 0, 0] * 8])

res = boi.forward(x, y)


print(res)

res = boi.backward(boi.output, y)

print(boi.dinputs)
