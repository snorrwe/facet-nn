from .pyfacet import scalar, zeros, sqrt


class SGD:
    def __init__(self, *, learning_rate, decay=0.0, momentum=None):
        self.initial_lr = learning_rate
        self.lr = scalar(learning_rate)
        self.decay = decay
        self.iters = 0
        if momentum is not None:
            self.momentum = scalar(momentum)
        else:
            self.momentum = momentum

    def pre_update(self):
        self.lr = scalar(self.initial_lr * (1.0 / (1 + self.decay * self.iters)))
        self.iters += 1

    def update_params(self, layer):
        # calculate deltas
        if self.momentum is not None:
            # SGD with momentum
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = zeros(layer.weights.shape)
                layer.bias_momentums = zeros(layer.biases.shape)

            nabla_w = self.momentum * layer.weight_momentums - self.lr * layer.dweights
            layer.weight_momentums = nabla_w
            nabla_b = self.momentum * layer.bias_momentums - self.lr * layer.dbiases
            layer.bias_momentums = nabla_b
        else:
            # vanilla SGD
            nabla_w = scalar(-1.0) * self.lr * layer.dweights
            nabla_b = scalar(-1.0) * self.lr * layer.dbiases

        # update params
        layer.weights += nabla_w
        layer.biases += nabla_b


class Adam:
    def __init__(
        self, *, learning_rate, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999
    ):
        self.initial_lr = learning_rate
        self.lr = scalar(learning_rate)
        self.decay = decay
        self.iters = 0
        self.epsilon = scalar(epsilon)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.weight_cache = {}
        self.bias_cache = {}
        self.weight_momentums = {}
        self.bias_momentums = {}

    def pre_update(self):
        self.lr = scalar(self.initial_lr * (1.0 / (1 + self.decay * self.iters)))
        self.iters += 1

    def update_params(self, layer):
        lid = layer.id
        if not hasattr(layer, "weight_cache"):
            self.weight_cache[lid] = zeros(layer.weights.shape)
            self.bias_cache[lid] = zeros(layer.biases.shape)
            self.weight_momentums[lid] = zeros(layer.weights.shape)
            self.bias_momentums[lid] = zeros(layer.biases.shape)

        self.weight_momentums[lid] = (
            scalar(self.beta_1) * self.weight_momentums[lid]
        ) + (scalar(1.0 - self.beta_1) * layer.dweights)
        self.bias_momentums[lid] = (scalar(self.beta_1) * self.bias_momentums[lid]) + (
            scalar(1.0 - self.beta_1) * layer.dbiases
        )

        beta1_inv = scalar( 1 - (self.beta_1 ** self.iters))
        beta2_inv = scalar(
            1 - self.beta_2
        )
        weight_momentum_corrected = self.weight_momentums[lid] / beta1_inv
        bias_momentum_corrected = self.bias_momentums[lid] / beta1_inv

        self.weight_cache[lid] = scalar(self.beta_2) * self.weight_cache[lid] + beta2_inv * (layer.dweights ** 2)
        self.bias_cache[lid] = scalar(self.beta_2) * self.bias_cache[lid] + beta2_inv * (layer.dbiases ** 2)

        weight_cache_corrected = self.weight_cache[lid] / scalar(
            1 - self.beta_2 ** self.iters
        )
        bias_cache_corrected = self.bias_cache[lid] / scalar(
            1 - self.beta_2 ** self.iters
        )

        layer.weights -= (
            self.lr
            * weight_momentum_corrected
            / (sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases -= (
            self.lr
            * bias_momentum_corrected
            / (sqrt(bias_cache_corrected) + self.epsilon)
        )
