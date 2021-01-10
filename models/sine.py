import progressbar

import pyfacet as pf
from pyfacet.optimizer import Adam

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


inp = pf.DenseLayer(1, 8)
acti1 = pf.Activation(pf.relu, pf.drelu_dz)
hidden = pf.DenseLayer(8, 8)
acti2 = pf.Activation(pf.relu, pf.drelu_dz)
dropout = pf.DropoutLayer(0.3)
out = pf.DenseLayer(8, 1)
acti3 = pf.Activation(pf.sigmoid, pf.dsigmoid)

loss = 0 # TODO rename + impl

optim = Adam(learning_rate=5e-4, decay=5e-5)

last = pf.scalar(0)
for epoch in progressbar.progressbar(range(250000 + 1), redirect_stdout=True):
    # forward pass
    inp.forward(X)
    acti1.forward(inp.output)
    hidden.forward(acti1.output)
    acti2.forward(hidden.output)
    dropout.forward(acti1.output)
    out.forward(dropout.output)
    acti3.forward(out.output)


    loss = loss.forward(acti3.output, y)[0]
    acc = pf.accuracy(loss.output, y)[0]

    if epoch % 1000 == 0:
        assert loss != last, "something's wrong i can feel it"
        last = loss
        lr = optim.lr[0]
        print(
            f"epoch {epoch:05} Loss: {loss:.16f} Accuracy: {acc:.16f} Learning Rate: {lr:.16f}"
        )
    # backward pass
    loss.backward(loss.output, y)
    acti3.backward(loss.output)
    out.backward(acti3.dinputs)
    dropout.backward(out.dinputs)
    acti2.backward(dropout.dinputs)
    hidden.backward(acti2.dinputs)
    acti1.backward(hidden.dinputs)
    inp.backward(hidden.dinputs)

    # optim
    optim.pre_update()
    optim.update_params(dense1)
    optim.update_params(dense2)
    optim.update_params(dense3)
