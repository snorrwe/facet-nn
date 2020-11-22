from nnfs import *
import nd


layer1 = DenseLayer(
    weights=nd.array(list(range(9))).reshape([3, 3]),
    biases=nd.array(list(range(4, 7))),
    activation=nd.softmax,
)

inp = nd.array([[1, 1, 1]] * 9)

out = layer1.forward(inp)

print(repr(out))


print("Loading data")

dataset = nd.load_csv("C:/Users/dkiss/Downloads/IRIS.csv", labels=["species"])

print(dataset["columns"])
for label, row in list(zip(dataset["labels"], dataset["data"].iter_cols()))[:5]:
    print(row, label)
