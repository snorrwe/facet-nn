import progressbar

import pyfacet as pf
from pyfacet.layer import DropoutLayer
from pyfacet.optimizer import Adam
from pyfacet.activation import ActivationLinear
from pyfacet.loss import MeanSquaredError

import random
import math


def generate(n):
    for _ in range(n):
        x = random.randrange(-100, 100, 1)
        x /= 100.0
        yield (x, math.sin(x))


X = []
Y = []

for (x, y) in generate(10000):
    X.append([x])
    Y.append([y])


X = pf.array(X)
Y = pf.array(Y)

# TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# changing layer sizes results in panic during mat mul

inp = pf.DenseLayer(1, 8)
acti1 = pf.Activation(pf.relu, pf.drelu_dz)
hidden = pf.DenseLayer(8, 8)
acti2 = pf.Activation(pf.relu, pf.drelu_dz)
dropout = DropoutLayer(0.3)
out = pf.DenseLayer(8, 1)
acti3 = ActivationLinear()

loss = MeanSquaredError()


optim = Adam(learning_rate=5e-4, decay=5e-5)

last = pf.scalar(0)
for epoch in progressbar.progressbar(range(1000 + 1), redirect_stdout=True):
    # forward pass
    inp.forward(X)
    acti1.forward(inp.output)
    hidden.forward(acti1.output)
    acti2.forward(hidden.output)
    dropout.forward(acti1.output)
    out.forward(dropout.output)
    acti3.forward(out.output)
    loss.forward(acti3.output, Y)

    data_loss = loss.calculate()
    acc = float("nan")  # TODO # old (does not work here): pf.accuracy(data_loss, Y)[0]
    data_loss = data_loss[0]

    if epoch % 100 == 0:
        assert data_loss != last, "something's wrong i can feel it"
        last = data_loss
        lr = optim.lr[0]
        print(
            f"epoch {epoch:05} Loss: {data_loss:.16f} Accuracy: {acc:.16f} Learning Rate: {lr:.16f}"
        )

    # backward pass
    loss.backward(acti3.output, Y)
    acti3.backward(loss.dinputs)
    out.backward(acti3.dinputs)
    dropout.backward(out.dinputs)
    acti2.backward(dropout.dinputs)
    hidden.backward(acti2.dinputs)
    acti1.backward(hidden.dinputs)
    inp.backward(acti1.dinputs)

    # optim
    optim.pre_update()
    optim.update_params(inp)
    optim.update_params(hidden)
    optim.update_params(out)
