from nnfs import *
import pydu


boi = Activation_Softmax_Loss_CategoricalCrossentropy()


x = pydu.array([[1, 2, 8]] * 8)
y = pydu.array([[1, 0, 0]] * 8)


dense1 = DenseLayer(3, 8)

res= dense1.forward(x)
dense1.backward(res)

optim = Optimizer_SGD(0.001)

optim.update_params(dense1)
