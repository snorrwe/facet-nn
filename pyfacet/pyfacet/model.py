# TODO: register dependency
import progressbar
from uuid import uuid4

from .layer import InputLayer


class Model:
    def __init__(self):
        self.layers = []
        self.prevlayer = {}
        self.nextlayer = {}
        self.trainable = []
        self.baked = False

    def add(self, layer):
        if not hasattr(layer, "id"):
            layer.id = uuid4()
        self.layers.append(layer)
        if hasattr(layer, "weights"):
            self.trainable.append(layer)

    def set(self, *, loss, optimizer, accuracy, input_layer=None):
        self.loss = loss
        self.optimizer = optimizer
        self.input_layer = input_layer if input_layer is not None else InputLayer()
        self.accuracy = accuracy
        if not hasattr(self.input_layer, "id"):
            self.input_layer.id = uuid4()

    def bake(self):
        """
        prepares the previously added layers for execution
        """
        if not self.layers:
            return

        self.loss.trainable_layers = self.trainable

        count = len(self.layers)
        lastid = self.layers[0].id
        self.prevlayer[lastid] = self.input_layer
        for i in range(1, count):
            l = self.layers[i]
            self.nextlayer[lastid] = l
            lastid = l.id
            self.prevlayer[lastid] = self.layers[i - 1]
        self.nextlayer[lastid] = self.loss
        self.output_activation = self.layers[i]
        self.loss.trainable_layers = self.trainable
        self.baked = True

    def forward(self, X):
        assert self.baked

        self.input_layer.forward(X)
        for l in self.layers:
            l.forward(self.prevlayer[l.id].output)
        return l.output

    def backward(self, output, y):
        self.loss.backward(output, y)
        for l in reversed(self.layers):
            l.backward(self.nextlayer[l.id].dinputs)

    def train(self, X, y, *, epochs=1, print_every=1):
        assert self.baked

        last = 0.0

        self.accuracy.init(y)
        for epoch in progressbar.progressbar(range(epochs + 1), redirect_stdout=True):
            output = self.forward(X)
            data_loss, reg_loss = self.loss.calculate(output, y)

            if print_every and epoch % print_every == 0:
                assert data_loss != last, "something's wrong i can feel it"
                last = data_loss
                lr = self.optimizer.lr[0]
                acc = self.accuracy.calculate(output, y)
                print(
                    f"epoch {epoch:05} Loss: {data_loss:.16f} Accuracy: {acc:.16f} Learning Rate: {lr:.16f}"
                )

            self.backward(output, y)

            self.optimizer.pre_update()
            for l in self.trainable:
                self.optimizer.update_params(l)
