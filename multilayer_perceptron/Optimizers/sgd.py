# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    sgd.py                                            :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:18:45 by cmariot          #+#    #+#              #
#    Updated: 2023/09/30 12:20:26 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np


class StandardGradientDescent:

    """
    Standard Gradient Descent (SGD) optimizer with decay and momentum.
    Decay is used to reduce the learning rate over time.
    Momentum is used to accelerate SGD in the relevant direction
    """

    def __init__(self, learning_rate, decay, momentum):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def update(self, layer):

        # SGD with momentum
        if self.momentum is not None:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            layer.weights += weight_updates
            layer.biases += bias_updates

        # Default SGD
        else:
            layer.weights -= self.current_learning_rate * layer.dweights
            layer.biases -= self.current_learning_rate * layer.dbiases

    def update_learning_rate(self):
        if self.decay is not None:
            self.current_learning_rate = \
                self.learning_rate * (1. / (1. + self.decay * self.iterations))
            self.iterations += 1