import nd
from nnfs import *

print("Loading data")

dataset = nd.load_csv("C:/Users/dkiss/Downloads/IRIS.csv", labels=["species"])

print("Dataset loaded")
print(dataset["columns"])
for label, row in list(zip(dataset["labels"], dataset["data"].iter_cols()))[:5]:
    print(row, label)


inplayer = DenseLayer(
    weights=nd.array(list(range(9))).reshape([3, 3]),
    biases=nd.array(list(range(4, 7))),
    activation=nd.relu,
)

n_classes = len(set(dataset["labels"]))

layers = [
    # input layer
    DenseLayer(
        weights=nd.array(list(range(4 * 8))).reshape([4, 8]),
        biases=nd.array(list(range(4, 4 + 8))),
        activation=nd.softmax,
    ),
    # hidden 1
    DenseLayer(
        weights=nd.array(list(range(8 * 8))).reshape([8, 8]),
        biases=nd.array(list(range(4, 4 + 8))),
        activation=nd.relu,
    ),
    # out
    DenseLayer(
        weights=nd.array(list(range(8 * n_classes))).reshape([8, n_classes]),
        biases=nd.array(list(range(n_classes))),
        activation=nd.softmax,
    ),
]


def forward(x, network):
    for layer in network:
        x = layer.forward(x)
    return x


y = forward(dataset["data"], layers)

print(y)
