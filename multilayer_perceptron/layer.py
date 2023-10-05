# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    layer.py                                          :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:13:17 by cmariot          #+#    #+#              #
#    Updated: 2023/10/01 11:45:05 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np
from multilayer_perceptron.Activation.relu import ReluActivation
from multilayer_perceptron.Activation.sigmoid import SigmoidActivation
from multilayer_perceptron.Activation.softmax import SoftmaxActivation


class Layer:
    """Dense layer"""

    def __init__(self, n_inputs, n_neurons, activation_function):
        """
        Layer condstructor
        Args :
        - n_inputs : Number of inputs (number of neurons in previous layer)
        - n_neurons : Number of neurons
        - activation_function : Activation function to use
        """

        # Weights and biases init
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

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
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
