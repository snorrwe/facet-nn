import pydu


class Optimizer_SGD:
    def __init__(self, *, learning_rate, decay=0.0, momentum=None):
        self.initial_lr = learning_rate
        self.lr = pydu.scalar(learning_rate)
        self.decay = decay
        self.iters = 0
        if momentum is not None:
            self.momentum = pydu.scalar(momentum)
        else:
            self.momentum = momentum

    def pre_update(self):
        self.lr = pydu.scalar(self.initial_lr * (1.0 / (1 + self.decay * self.iters)))
        self.iters += 1

    def update_params(self, layer):
        # calculate deltas
        if self.momentum is not None:
            # SGD with momentum
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = pydu.zeros(layer.weights.shape)
                layer.bias_momentums = pydu.zeros(layer.biases.shape)

            nabla_w = self.momentum * layer.weight_momentums - self.lr * layer.dweights
            layer.weight_momentums = nabla_w
            nabla_b = self.momentum * layer.bias_momentums - self.lr * layer.dbiases
            layer.bias_momentums = nabla_b
        else:
            # vanilla SGD
            nabla_w = pydu.scalar(-1.0) * self.lr * layer.dweights
            nabla_b = pydu.scalar(-1.0) * self.lr * layer.dbiases

        # update params
        layer.weights += nabla_w
        layer.biases += nabla_b


class Optimizer_Adam:
    def __init__(
        self, *, learning_rate, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999
    ):
        self.initial_lr = learning_rate
        self.lr = pydu.scalar(learning_rate)
        self.decay = decay
        self.iters = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update(self):
        self.lr = pydu.scalar(self.initial_lr * (1.0 / (1 + self.decay * self.iters)))
        self.iters += 1

    def update_params(self, layer):
        pass
