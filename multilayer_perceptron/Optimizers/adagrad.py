import numpy as np


class AdaGrad:

    def __init__(self, learning_rate, decay, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

        if epsilon is None:
            epsilon = 1e-7
        self.epsilon = epsilon

    def update(self, layer):

        try:

            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            layer.weight_cache += layer.dweights ** 2
            layer.bias_cache += layer.dbiases ** 2

            layer.weights -= self.current_learning_rate * layer.dweights / \
                (np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases -= self.current_learning_rate * layer.dbiases / \
                (np.sqrt(layer.bias_cache) + self.epsilon)

        except Exception as error:
            print("Error: can't update the layer,", error)
            exit()

    def update_learning_rate(self):
        if self.decay is not None:
            self.current_learning_rate = \
                self.learning_rate * (1. / (1. + self.decay * self.iterations))
            self.iterations += 1
