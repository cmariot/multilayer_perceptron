# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    layer.py                                          :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:13:17 by cmariot          #+#    #+#              #
#    Updated: 2023/09/28 15:07:15 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np
from multilayer_perceptron.Activation.relu import ReluActivation
from multilayer_perceptron.Activation.sigmoid import SigmoidActivation
from multilayer_perceptron.Activation.softmax import SoftmaxActivation


class Layer:

    # Available activation functions
    activation_functions = {
        "relu": ReluActivation,
        "sigmoid": SigmoidActivation,
        "softmax": SoftmaxActivation
    }

    def __init__(self,
                 n_inputs,
                 n_neurons,
                 activation_function
                 ):
        """
        Layer condstructor
        """

        # Weights and biases init
        self.weights = \
            (np.sqrt(2.0 / n_inputs)) * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Momentums : for momentum optimizer
        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.biases)

        # Activation function init
        if activation_function not in self.activation_functions:
            raise Exception("Activation function not found")
        self.activation_function = \
            self.activation_functions[activation_function]()

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
