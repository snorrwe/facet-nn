import csv

import progressbar

import pydu
from nnfs import *

print("Loading data")


def normalized_data(rowdict):
    try:
        fare = float(rowdict["Fare"]) / 50
    except:
        fare = 0
    return [
        1 if rowdict["Sex"] == "female" else -1,
        float(rowdict["Pclass"]) - 2,
        (float(rowdict["Age"]) if rowdict["Age"] else 0) / 80,
        fare,
    ]


def load_data(fname):
    labels = []
    data = []
    meta = []
    with open(fname) as f:
        for row in csv.DictReader(f):
            data.append(normalized_data(row))
            meta.append(row["PassengerId"])
            if "Survived" in row:
                labels.append(int(row["Survived"]))
    if labels:
        y = labels_to_y(labels)
    else:
        y = None
    X = pydu.array(data)
    return (X, y, meta)


(X, y, _) = load_data("C:/Users/dkiss/Downloads/titanic/train.csv")

n_classes = 2
n_inputs = X.shape[1]

# the model
dense1 = DenseLayer(n_inputs, 8)
acti1 = Activation(pydu.relu, pydu.drelu_dz)

dense2 = DenseLayer(8, 13)
acti2 = Activation(pydu.relu, pydu.drelu_dz)

dense3 = DenseLayer(13, n_classes)
loss_acti = Activation_Softmax_Loss_CategoricalCrossentropy()


optim = Optimizer_Adam(learning_rate=5e-3, decay=1e-4)

last = pydu.scalar(0)
for epoch in progressbar.progressbar(range(10000 + 1), redirect_stdout=True):
    dense1.forward(X)
    acti1.forward(dense1.output)
    dense2.forward(acti1.output)
    acti2.forward(dense2.output)
    dense3.forward(acti2.output)

    loss = loss_acti.forward(dense3.output, y)[0]
    acc = accuracy(loss_acti.output, y)[0]

    if epoch % 1000 == 0:
        assert loss != last, "something's wrong i can feel it"
        last = loss
        lr = optim.lr[0]
        print(
            f"epoch {epoch:05} Loss: {loss:.16f} Accuracy: {acc:.16f} Learning Rate: {lr:.16f}"
        )
    # backward pass
    loss_acti.backward(loss_acti.output, y)
    dense3.backward(loss_acti.dinputs)
    acti2.backward(dense3.dinputs)
    dense2.backward(acti2.dinputs)
    acti1.backward(dense2.dinputs)
    dense1.backward(acti1.dinputs)

    # optim
    optim.pre_update()
    optim.update_params(dense1)
    optim.update_params(dense2)
    optim.update_params(dense3)


print("Training done, validating")
print(
    f"epoch {epoch:05} Loss: {loss:.16f} Accuracy: {acc:.16f} Learning Rate: {lr:.16f}"
)

(X, _, meta) = load_data("C:/Users/dkiss/Downloads/titanic/test.csv")
dense1.forward(X)
acti1.forward(dense1.output)
dense2.forward(acti1.output)
acti2.forward(dense2.output)
dense3.forward(acti2.output)

finalacti = Activation(pydu.softmax)
finalacti.forward(dense3.output)

predi = pydu.argmax(finalacti.output)

# next(iter_cols) is a list of the items in the cols == the items in the array
output = zip(meta, next(predi.iter_cols()))
with open("submission.csv", "w") as f:
    writer = csv.DictWriter(f, ["PassengerId", "Survived"])
    writer.writeheader()
    for row in output:
        writer.writerow({"PassengerId": row[0], "Survived": row[1]})
