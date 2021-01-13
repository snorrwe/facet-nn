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

    def add(self, layer):
        if not hasattr(layer, "id"):
            layer.id = uuid4()
        self.layers.append(layer)
        if hasattr(layer, "weights"):
            self.trainable.append(layer)

    def set(self, *, loss, optimizer, input_layer=None):
        self.loss = loss
        self.optimizer = optimizer
        self.input_layer = input_layer if input_layer is not None else InputLayer()
        if not hasattr(self.input_layer, "id"):
            self.input_layer.id = uuid4()

    def bake(self):
        """
        prepares the previously added layers for execution
        """
        if not self.layers:
            return

        count = len(self.layers)
        lastid = self.layers[0].id
        self.prevlayer[lastid] = self.input_layer
        for i in range(1, count):
            l = self.layers[i]
            self.nextlayer[lastid] = l
            lastid = l.id
            self.prevlayer[lastid] = self.layers[i - 1]
        self.nextlayer[lastid] = self.loss

    def forward(self, X):
        self.input_layer.forward(X)
        for i, l in enumerate(self.layers):
            l.forward(self.prevlayer[l.id].output)
        return l.output

    def train(self, X, y, *, epochs=1, print_every=1):
        for epoch in progressbar.progressbar(range(epochs + 1), redirect_stdout=True):
            output = self.forward(X)
            # TODO
            assert 0, "TODO %s" % output
            if epoch % print_every == 0:
                assert data_loss != last, "something's wrong i can feel it"
                last = data_loss
                lr = optim.lr[0]
                print(
                    f"epoch {epoch:05} Loss: {data_loss:.16f} Accuracy: {acc:.16f} Learning Rate: {lr:.16f}"
                )
