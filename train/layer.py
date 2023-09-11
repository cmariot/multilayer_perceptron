import numpy as np
from single_neuron import SingleNeuron


class Layer:

    def __init__(self, weights, biases, inputs):
        self.weights = weights
        self.biases = biases
        self.inputs = inputs


    def iterative_forward(self, neurons):
        """
        Unoptimized forward pass for a layer of neurons
        The output of a neuron is the sum of the weighted inputs plus the bias
        """
        self.output = []
        for neuron in neurons:
            self.output.append(neuron.forward())
        return self.output

    def forward(self):
        self.output = np.dot(self.weights, self.inputs) + self.biases
        return self.output




if __name__ == "__main__":

    # ################ #
    # Layer of neurons #
    # ################ #

    # Input of the layer, each neuron will have the same input, a sample
    # In this case, we have 4 features for each sample
    inputs = np.array([1, 2, 3, 2.5])

    # If we create a layer of 3 neurons, each neuron will have 4 weights and 1 bias,
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
        None
    )

    # Forward pass, we get the output of each neuron
    layer.iterative_forward([neuron1, neuron2, neuron3])
    print(layer.output)

    weights = np.array([
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ])

    biases = np.array([2, 3, 0.5])

    layer = Layer(
        inputs=inputs,
        weights=weights,
        biases=biases
    )

    layer.forward()

    print(layer.output)


