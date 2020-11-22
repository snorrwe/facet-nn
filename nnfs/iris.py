import nd
from nnfs import *

print("Loading data")

dataset = nd.load_csv("C:/Users/dkiss/Downloads/IRIS.csv", labels=["species"])

print("Dataset loaded")
print(dataset["columns"])
for label, row in first_n(5, zip(dataset["labels"], dataset["data"].iter_cols())):
    print(row, label)

n_classes = len(set(dataset["labels"]))

nw = Network(
    [
        DenseLayer(4, 8, activation=nd.softmax),  # input layer
        DenseLayer(8, 8, activation=nd.relu),  # hidden 1
        DenseLayer(8, n_classes, activation=nd.softmax),  # out
    ]
)


pred = nw.forward(dataset["data"])


l = Loss(nd.categorical_cross_entropy)

y = labels_to_y(dataset["labels"])

err = l.calculate(pred, y)

acc = accuracy(pred, y)

print(err, acc)
