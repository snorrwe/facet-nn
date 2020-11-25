import pydu
from nnfs import *
import progressbar

DEBUG = False

print("Loading data")

dataset = pydu.load_csv("C:/Users/dkiss/Downloads/iris.csv", labels=["Species"])

print("Dataset loaded")
print(dataset["columns"])
for label, row in first_n(5, zip(dataset["labels"], dataset["data"].iter_cols())):
    print(row, label)
    n_inputs = len(row)

n_classes = len(set(dataset["labels"]))

INNER = 3

dense1 = DenseLayer(n_inputs, INNER, name="dense1")
acti1 = Activation(pydu.relu, pydu.drelu_dz)

dense2 = DenseLayer(INNER, n_classes, name="dense2")
loss_acti = Activation_Softmax_Loss_CategoricalCrossentropy()

optim = Optimizer_SGD(0.001)

y = labels_to_y(dataset["labels"])
print("Lets fucking go")
last = pydu.scalar(0)
for epoch in progressbar.progressbar(range(10000 + 1), redirect_stdout=True):
    dense1.forward(dataset["data"])
    acti1.forward(dense1.output)
    dense2.forward(acti1.output)
    loss = loss_acti.forward(dense2.output, y)
    acc = accuracy(loss_acti.output, y)

    if epoch % 100 == 0:
        assert (loss != last).all(), "somethings wrong i can feel it"
        last = loss
        print(f"epoch {epoch} Loss: {loss} Accuracy: {acc}")

    # backward pass
    loss_acti.backward(loss_acti.output, y)
    dense2.backward(loss_acti.dinputs)
    acti1.backward(dense2.dinputs)
    dense1.backward(acti1.dinputs)

    # update weights & biases
    optim.update_params(dense1)
    optim.update_params(dense2)

    if DEBUG:
        break
