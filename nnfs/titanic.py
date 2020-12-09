import csv

import progressbar

import pyfacet as pyf
from pyfacet.optimizers import Adam
from pyfacet import DenseLayer, Activation, DropoutLayer, accuracy

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
        y = pyf.labels_to_y(labels)
    else:
        y = None
    X = pyf.array(data)
    return (X, y, meta)


(X, y, _) = load_data("C:/Users/dkiss/Downloads/titanic/train.csv")

n_classes = 2
n_inputs = X.shape[1]

# the model
dense1 = DenseLayer(
    n_inputs,
    17,
    weight_regularizer_l1=0.1,
    weight_regularizer_l2=5e-4,
    bias_regularizer_l2=5e-4,
)
acti1 = Activation(pyf.relu, pyf.drelu_dz)

dropout1 = DropoutLayer(0.3)

dense2 = DenseLayer(17, 13)
acti2 = Activation(pyf.relu, pyf.drelu_dz)

dropout2 = DropoutLayer(0.1)

dense3 = DenseLayer(13, n_classes)
loss_acti = pyf.Activation_Softmax_Loss_CategoricalCrossentropy()

optim = Adam(learning_rate=5e-4, decay=5e-5)

last = pyf.scalar(0)
for epoch in progressbar.progressbar(range(15000 + 1), redirect_stdout=True):
    dense1.forward(X)
    acti1.forward(dense1.output)
    dropout1.forward(acti1.output)
    dense2.forward(dropout1.output)
    acti2.forward(dense2.output)
    dropout2.forward(acti2.output)
    dense3.forward(dropout2.output)

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
    dropout2.backward(dense3.dinputs)
    acti2.backward(dropout2.dinputs)
    dense2.backward(acti2.dinputs)
    dropout1.backward(dense2.dinputs)
    acti1.backward(dropout1.dinputs)
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

finalacti = Activation(pyf.softmax)
finalacti.forward(dense3.output)

predi = pyf.argmax(finalacti.output)

output = zip(meta, iter(predi))
with open("submission.csv", "w") as f:
    writer = csv.DictWriter(f, ["PassengerId", "Survived"])
    writer.writeheader()
    for row in output:
        writer.writerow({"PassengerId": row[0], "Survived": row[1]})
