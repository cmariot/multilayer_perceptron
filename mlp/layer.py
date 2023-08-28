# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    layer.py                                          :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:39 by cmariot          #+#    #+#              #
#    Updated: 2023/08/24 19:50:53 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np
from ActivationFunctions.sigmoid import Sigmoid_Activation
from ActivationFunctions.softmax import Softmax_Activation


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
                 learning_rate,
                 decay,
                 momentum):
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
            self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
            self.biases = np.ones((1, n_neurons))
            self.activation = self.activation_function[activation]()
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            self.decay = decay
            self.iterations = 0
            self.momentum = momentum
            self.weight_momentums = np.ones(self.weights.shape)
            self.bias_momentums = np.ones(self.biases.shape)
        except Exception as e:
            print("Error (init Dense_Layer) :", e)
            exit()

    def update_learning_rate(self):
        if self.decay > 0.0:
            self.current_learning_rate = self.learning_rate * \
                (1.0 / (1.0 + self.decay * self.iterations))

    def forward(self, inputs):
        self.inputs = inputs
        self.weighted_sum = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.forward(self.weighted_sum)
        return self.output

    def update_iterations(self):
        self.iterations += 1

    def backward(self, dvalues):
        dvalues = self.activation.backward(dvalues)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

    def gradient_descent(self):
        weight_updates = self.momentum * self.weight_momentums - \
            self.current_learning_rate * self.dweights
        self.weight_momentums = weight_updates

        bias_updates = self.momentum * self.bias_momentums - \
            self.current_learning_rate * self.dbiases
        self.bias_momentums = bias_updates

        self.weights += weight_updates
        self.biases += bias_updates
