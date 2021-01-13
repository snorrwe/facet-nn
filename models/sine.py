import progressbar

import pyfacet as pf
from pyfacet.layer import DropoutLayer
from pyfacet.optimizer import Adam
from pyfacet.activation import ActivationLinear, Activation
from pyfacet.loss import MeanSquaredError
from pyfacet.model import Model

import random
import math


def generate(n):
    for _ in range(n):
        x = random.randrange(-31415, 31415, 1)
        x /= 10000.0
        yield (x, math.sin(x))


X = []
Y = []

for (x, y) in generate(10):
#  for (x, y) in generate(10000):
    X.append([x])
    Y.append([y])


X = pf.array(X)
Y = pf.array(Y)


loss = MeanSquaredError()

model = Model()
model.add(pf.DenseLayer(1, 64))
model.add(Activation(pf.relu, pf.drelu_dz))
model.add(pf.DenseLayer(64, 64))
model.add(Activation(pf.relu, pf.drelu_dz))
model.add(DropoutLayer(0.3))
model.add(pf.DenseLayer(64, 1))
model.add(ActivationLinear())

model.set(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1e-2, decay=1e-3))

model.bake()

model.train(X, Y, epochs=10000, print_every=100)

#  last = pf.scalar(0)
#  for epoch in progressbar.progressbar(range(10000 + 1), redirect_stdout=True):
#      # forward pass
#      inp.forward(X)
#      acti1.forward(inp.output)
#      hidden.forward(acti1.output)
#      acti2.forward(hidden.output)
#      dropout.forward(acti2.output)
#      out.forward(dropout.output)
#      acti3.forward(out.output)
#      loss.forward(acti3.output, Y)
#
#      data_loss = loss.calculate()
#      acc = float("nan")  # TODO # old (does not work here): pf.accuracy(data_loss, Y)[0]
#      data_loss = data_loss[0]
#
#      if epoch % 100 == 0:
#          assert data_loss != last, "something's wrong i can feel it"
#          last = data_loss
#          lr = optim.lr[0]
#          print(
#              f"epoch {epoch:05} Loss: {data_loss:.16f} Accuracy: {acc:.16f} Learning Rate: {lr:.16f}"
#          )
#
#      # backward pass
#      loss.backward(acti3.output, Y)
#      acti3.backward(loss.dinputs)
#      out.backward(acti3.dinputs)
#      dropout.backward(out.dinputs)
#      acti2.backward(dropout.dinputs)
#      hidden.backward(acti2.dinputs)
#      acti1.backward(hidden.dinputs)
#      inp.backward(acti1.dinputs)
#
#      # optim
#      optim.pre_update()
#      optim.update_params(inp)
#      optim.update_params(hidden)
#      optim.update_params(out)


with open("out.txt", "w") as f:
    for x, actual, expected in zip(X, out.output, Y):
        f.write("{} {} {}\n".format(x, actual, expected))
