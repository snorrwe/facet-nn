import pyfacet as pf
from pyfacet.optimizer import Adam
from pyfacet.layer import DropoutLayer, DenseLayer
from pyfacet.model import Model
from pyfacet.activation import ActivationLinear, Activation
from pyfacet.loss import Loss
from pyfacet import categorical_cross_entropy, cce_backward
from pyfacet.accuracy import Accuracy_Categorical


dataset = pf.load_csv(
    "c:/users/dkiss/downloads/digit-recognizer/train.csv",
    labels=["label"],
)

X = dataset["data"]
y = pf.labels_to_one_hot(dataset["labels"])

model = Model()

model.add(pf.DenseLayer(X.shape[1], 32))
model.add(Activation(pf.relu, df=pf.drelu_dz))
model.add(pf.DenseLayer(32, 100, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation(pf.relu, df=pf.drelu_dz))
model.add(DropoutLayer(0.1))
model.add(pf.DenseLayer(100, 10))
model.add(Activation(pf.softmax, df=pf.dsoftmax))

model.set(
    loss=Loss(lossfn=categorical_cross_entropy, dlossfn=cce_backward),
    accuracy=Accuracy_Categorical(),
    optimizer=Adam(learning_rate=1e-4, decay=1e-4),
)


model.bake()

print(X.shape)
model.train(X, y, epochs=10, print_every=1)
