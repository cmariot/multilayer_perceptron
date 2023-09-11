import numpy as np


class SingleNeuron:

    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.biases = bias

    def iterative_forward(self):
        """
        Unoptimized forward pass
        The output of a neuron is the sum of the weighted inputs plus the bias
        """
        self.output = 0.
        for input, weight in zip(self.inputs, self.weights):
            self.output += input * weight
        self.output += self.biases
        return self.output

    def forward(self):
        """
        Optimized forward pass
        Same as iterative_forward but using numpy
        """
        self.output = np.dot(self.inputs, self.weights) + self.biases
        return self.output


if __name__ == "__main__":

    # ############# #
    # Single neuron #
    # ############# #

    # Input of the neuron
    inputs = np.array([1, 2, 3])

    # Weights and bias associated to the neuron
    weights = np.array([0.2, 0.8, -0.5])
    bias = 2

    # Create the neuron
    neuron = SingleNeuron(
        inputs=inputs,
        weights=weights,
        bias=bias
    )

    # Iterative forward pass
    neuron.iterative_forward()
    print(neuron.output)

    # Optimized forward pass
    neuron.forward()
    print(neuron.output)