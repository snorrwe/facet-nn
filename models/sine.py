import progressbar

import pyfacet as pf
from pyfacet.layer import DropoutLayer
from pyfacet.optimizer import Adam
from pyfacet.activation import ActivationLinear, Activation
from pyfacet.loss import MeanSquaredError
from pyfacet.model import Model
from pyfacet.accuracy import Accuracy_Regression

import random
import math


def generate(n):
    for _ in range(n):
        x = random.uniform(-3.1415, 3.1415)
        yield (x, math.sin(x))


X = []
Y = []

for (x, y) in generate(10000):
    X.append([x])
    Y.append([y])


X = pf.array(X)
Y = pf.array(Y)


model = Model()
model.add(pf.DenseLayer(1, 64))
model.add(Activation(pf.relu, df=pf.drelu_dz))
model.add(pf.DenseLayer(64, 64))
#  model.add(DropoutLayer(0.2))
model.add(Activation(pf.relu, df=pf.drelu_dz))
model.add(pf.DenseLayer(64, 1))
model.add(ActivationLinear())

for l in model.layers:
    if hasattr(l, 'weights'):
        l.weights *= pf.scalar(0.1) # flatten the weights
        l.biases = pf.array([0.0]*len(l.biases))

model.set(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=1e-3, decay=1e-3),
    accuracy=Accuracy_Regression(),
)

model.bake()

model.train(X, Y, epochs=10000, print_every=100)

with open("out.txt", "w") as f:
    for x, actual, expected in zip(X, model.layers[-1].output, Y):
        f.write("{} {} {}\n".format(x, actual, expected))
