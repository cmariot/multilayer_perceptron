# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multi_layer_perceptron.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:36 by cmariot           #+#    #+#              #
#    Updated: 2023/08/24 14:39:37 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

class MultiLayerPerceptron:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        """
        Forward propagation.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, dinput):
        """
        Backward propagation.
        """
        for layer in reversed(self.layers):
            dinput = layer.backward(dinput)
        return dinput

    def update_learning_rate(self):
        """
        Update learning rate.
        """
        for layer in self.layers:
            layer.update_learning_rate()

    def update_parameters(self):
        """
        Update parameters.
        """
        for layer in self.layers:
            layer.update()

    def update_iterations(self):
        """
        Update iterations.
        """
        for layer in self.layers:
            layer.update_iterations()
