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
y = []

for (a, b) in generate(10000):
    X.append([a])
    y.append([b])

X_val = []
y_val = []
for (a, b) in generate(100):
    X_val.append([a])
    y_val.append([b])


X = pf.array(X)
y = pf.array(y)
X_val = pf.array(X_val)
y_val = pf.array(y_val)


model = Model()
model.add(pf.DenseLayer(1, 128))
model.add(Activation(pf.relu, df=pf.drelu_dz))
model.add(pf.DenseLayer(128, 64))
#  model.add(DropoutLayer(0.2))
model.add(Activation(pf.relu, df=pf.drelu_dz))
model.add(pf.DenseLayer(64, 1))
model.add(ActivationLinear())

for l in model.layers:
    if hasattr(l, "weights"):
        l.weights *= pf.scalar(0.1)  # flatten the weights

model.set(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=1e-3, decay=1e-3),
    accuracy=Accuracy_Regression(),
)

model.bake()

model.train(X, y, epochs=10000, print_every=100, validation=(X_val, y_val))

output = model.forward(X_val)

with open("out.txt", "w") as f:
    for x, actual, expected in zip(X_val, output, y_val):
        f.write("{} {} {}\n".format(x, actual, expected))
