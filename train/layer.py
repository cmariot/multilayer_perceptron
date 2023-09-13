import numpy as np
from single_neuron import SingleNeuron
from nnfs.datasets import spiral_data


class Layer:

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def iterative_forward(self, neurons):
        """
        Unoptimized forward pass for a layer of neurons
        The output of a neuron is the sum of the weighted inputs plus the bias
        """
        self.output = []
        for neuron in neurons:
            self.output.append(neuron.forward())
        return self.output

    def forward(self, input):
        self.output = np.dot(input, self.weights.T) + self.biases
        return self.output


if __name__ == "__main__":

    # ################ #
    # Layer of neurons #
    # ################ #

    # Input of the layer, each neuron will have the same input,
    # In this case, we have 1 sample with 4 features
    inputs = np.array([1, 2, 3, 2.5])

    # If we create a layer of 3 neurons,
    # each neuron will have 4 weights (one for each feature) and 1 bias,
    # the weights and bias are different for each neuron

    neuron1 = SingleNeuron(
        inputs=inputs,
        weights=np.array([0.2, 0.8, -0.5, 1.0]),
        bias=2
    )

    neuron2 = SingleNeuron(
        inputs=inputs,
        weights=np.array([0.5, -0.91, 0.26, -0.5]),
        bias=3
    )

    neuron3 = SingleNeuron(
        inputs=inputs,
        weights=np.array([-0.26, -0.27, 0.17, 0.87]),
        bias=0.5
    )

    layer = Layer(
        None,
        None,
    )

    # Forward pass, we get the output of each neuron
    layer.iterative_forward([neuron1, neuron2, neuron3])
    print(layer.output)

    # weights = np.array([
    #     [0.2, 0.8, -0.5, 1.0],
    #     [0.5, -0.91, 0.26, -0.5],
    #     [-0.26, -0.27, 0.17, 0.87]
    # ])

    # biases = np.array([2, 3, 0.5])

    # layer = Layer(
    #     inputs=inputs,
    #     weights=weights,
    #     biases=biases
    # )

    # layer.forward()

    # print(layer.output)

    # Input data with more than one sample : batch
    batch_input = np.array(
        [
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]
        ]
    )

    weights = np.array(
        [
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87],
        ]
    )

    biases = np.array([2.0, 3.0, 0.5])

    layer.weights = weights
    layer.biases = biases

    layer.forward(batch_input)

    print(layer.output)

    # The output of the first layer will be the input of the second layer

    second_layer_input = layer.output

    weights = np.array(
        [
            [0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13],
        ]
    )

    biases = np.array([-1.0, 2.0, -0.5])

    second_layer = Layer(
        weights=weights,
        biases=biases
    )

    second_layer.forward(second_layer_input)

    print(second_layer.output)

    import matplotlib.pyplot as plt

    X, y = spiral_data(samples=100, classes=3)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.show()
