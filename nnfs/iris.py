import pydu
from nnfs import *

print("Loading data")

dataset = pydu.load_csv("C:/Users/dkiss/Downloads/IRIS.csv", labels=["species"])

print("Dataset loaded")
print(dataset["columns"])
for label, row in first_n(5, zip(dataset["labels"], dataset["data"].iter_cols())):
    print(row, label)
    n_inputs = len(row)

n_classes = len(set(dataset["labels"]))

nw = Network(
    [
        DenseLayer(n_inputs, 8, activation=pydu.softmax),  # input layer
        DenseLayer(8, 8, activation=pydu.relu),  # hidden 1
        DenseLayer(8, n_classes, activation=pydu.softmax),  # output layer
    ]
)


pred = nw.forward(dataset["data"])


l = Loss(pydu.categorical_cross_entropy)

y = labels_to_y(dataset["labels"])

err = l.calculate(pred, y)

acc = accuracy(pred, y)

print(err, acc)
