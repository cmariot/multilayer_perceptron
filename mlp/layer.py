import numpy as np
from ActivationFunctions.sigmoid import Sigmoid_Activation
from ActivationFunctions.softmax import Softmax_Activation


np.random.seed(42)


class Dense_Layer:
    """
    A dense layer is a layer where each perceptron is connected to
    every perceptrons of the next layer, which means that its output
    value becomes an input for the next neurons.
    """

    activation_function = {
        "sigmoid": Sigmoid_Activation,
        "softmax": Softmax_Activation
    }

    def __init__(self,
                 n_inputs,
                 n_neurons,
                 activation,
                 learning_rate):
        """
        n_inputs: number of inputs of the layer.
        n_neurons: number of neurons of the layer.
        activation: activation function to use.

        weights: weights of the layer.
        biais: biais of the layer.
        activation: activation function to use.

        Both weights and biais are initialized with random values,
        They will be updated during the training.
        """

        try:
            self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
            self.biases = np.zeros((1, n_neurons))
            self.activation = self.activation_function[activation]()
            self.learning_rate = learning_rate
        except Exception as e:
            print("Error (init Dense_Layer) :", e)
            exit()

    def forward(self, inputs):
        self.inputs = inputs
        self.weighted_sum = np.dot(inputs, self.weights) + self.biases

    def activation_forward(self):
        self.activation.forward(self.weighted_sum)
        self.output = self.activation.output

    def backward(self, dvalues):
        dvalues = self.activation.backward(dvalues)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

    def update(self):
        self.weights -= self.learning_rate * self.dweights
        self.biases -= self.learning_rate * self.dbiases
