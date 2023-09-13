import numpy as np
from nnfs.datasets import spiral_data


class DenseLayer:

    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases

        # np.random.randn produces a Gaussian distribution with
        # a mean of 0 and a variance of 0.01

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        # The output is a vector with a size equals to the number of
        # neurons in the layer.
        # The weighted sum is computed on the entire layer with np.dot, and
        # we add the biases to get the output.
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


if __name__ == "__main__":

    input, target = spiral_data(samples=100, classes=3)
    layer = DenseLayer(n_inputs=2, n_neurons=3)

    layer.forward(inputs=input)

    print(layer.output[:5])
