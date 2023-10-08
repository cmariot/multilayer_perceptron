# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    layer.py                                          :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:13:17 by cmariot          #+#    #+#              #
#    Updated: 2023/10/05 21:20:18 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np
from multilayer_perceptron.Activation.relu import ReluActivation
from multilayer_perceptron.Activation.sigmoid import SigmoidActivation
from multilayer_perceptron.Activation.softmax import SoftmaxActivation


class Layer:

    """Dense layer"""

    def __init__(
        self,
        n_neurons: int,           # Nb of neurons in the layer
        n_inputs: int,            # Nb of inputs (n_neurons in previous layer)
        activation_function: str  # Activation function to use
    ):

        # Seed init
        np.random.seed(42)

        # Weights and dweights init
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.dweights = np.zeros((n_neurons, n_inputs))

        # Biases and dbiases init
        self.biases = np.zeros((n_neurons, 1))
        self.dbiases = np.zeros((n_neurons, 1))

        # Activation function init
        activation_functions = {
            "relu": ReluActivation,
            "sigmoid": SigmoidActivation,
            "softmax": SoftmaxActivation
        }
        if activation_function not in activation_functions:
            raise Exception("Activation function not found")
        self.activation_function = activation_functions[activation_function]()

    def forward(self, input):

        try:

            self.input = input
            return np.dot(self.weights, input) + self.biases

        except Exception as error:
            print("Error: can't forward the layer,", error)
            exit()

    def backward(self, gradient):

        try:

            self.dweights = np.dot(gradient, self.input.T)
            self.dbiases = np.sum(gradient, axis=1, keepdims=True)
            return np.dot(self.weights.T, gradient)

        except Exception:
            print("Error: can't backward the layer")
            exit()
