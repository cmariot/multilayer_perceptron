import numpy as np
from single_neuron import SingleNeuron
from nnfs.datasets import spiral_data
from Activation.linear import LinearActivation
from Activation.relu import ReluActivation
from Activation.sigmoid import SigmoidActivation
from Activation.softmax import SoftmaxActivation
from Activation.step import StepActivation
from Optimizers.sgd import StandardGradientDescent


class Layer:

    # Available activation functions
    activation_functions = {
        "linear": LinearActivation,
        "relu": ReluActivation,
        "sigmoid": SigmoidActivation,
        "softmax": SoftmaxActivation,
        "sigmoid": SigmoidActivation,
        "step": StepActivation
    }

    # Available optimizers
    optimizers = {
        "sgd": StandardGradientDescent 
    }

    def __init__(self,
                 n_inputs,
                 n_neurons,
                 activation_function,
                 optimizer="sgd",
                 learning_rate=0.0001,
                 decay=0.0,
                 momentum=0.0
                 ):
        """
        Layer condstructor
        """

        # Weights and biases init
        self.weights = (np.sqrt(2.0 / n_inputs)) * np.random.randn(n_inputs, n_neurons)
        self.biases = np.ones((1, n_neurons))

        # Momentums : for momentum optimizer
        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.biases)

        # Activation function init
        if activation_function not in self.activation_functions:
            raise Exception("Activation function not found")
        self.activation_function = self.activation_functions[activation_function]()

        # Optimizer init
        if optimizer not in self.optimizers:
            raise Exception("Activation function not found")
        self.optimizer = self.optimizers[optimizer](learning_rate, decay, momentum)


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
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        # self.output = self.activation_function.forward(self.weighted_sum)
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

    def update(self):
        self.optimizer.pre_update_params()
        self.optimizer.update(self)
        self.optimizer.post_update_params()


if __name__ == "__main__":

    # ################ #
    # Layer of neurons #
    # ################ #

    # Input of the layer, each neuron will have the same input,
    # In this case, we have 1 sample with 4 features
    # inputs = np.array([1, 2, 3, 2.5])

    # If we create a layer of 3 neurons,
    # each neuron will have 4 weights (one for each feature) and 1 bias,
    # the weights and bias are different for each neuron

    # neuron1 = SingleNeuron(
    #     inputs=inputs,
    #     weights=np.array([0.2, 0.8, -0.5, 1.0]),
    #     bias=2
    # )

    # neuron2 = SingleNeuron(
    #     inputs=inputs,
    #     weights=np.array([0.5, -0.91, 0.26, -0.5]),
    #     bias=3
    # )

    # neuron3 = SingleNeuron(
    #     inputs=inputs,
    #     weights=np.array([-0.26, -0.27, 0.17, 0.87]),
    #     bias=0.5
    # )

    # layer = Layer(
    #     n_inputs=3,
    #     n_neurons=4,
    #     activation_function="relu",
    #     optimizer="sgd"
    # )

    # Forward pass, we get the output of each neuron
    # layer.iterative_forward([neuron1, neuron2, neuron3])
    # print(layer.output)

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

    layer = Layer(
        n_input=3,
        n_neurons=3
    )

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
        biases=biases,
        activation_function="softmax",
        optimizer="sgd"
    )

    second_layer.forward(second_layer_input)

    print(second_layer.output)

    import matplotlib.pyplot as plt

    X, y = spiral_data(samples=100, classes=3)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.show()
