# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    adam.py                                           :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/10/09 12:33:02 by cmariot          #+#    #+#              #
#    Updated: 2023/10/09 12:33:03 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np


class Adam:

    """
    Adam (Adaptive Momentum) optimizer.
    Instead of adapting the parameters based only on the gradient,
    Adam has the momentum term that accumulates the gradient over time and
    a per-parameter learning rate that is adapted based on the cache of
    the gradients.
    """

    def __init__(
        self,
        learning_rate,
        decay,
        epsilon=1e-7,
        momentum_coeff=0.9,
        cache_coeff=0.999,
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = 1e-7 if epsilon is None else epsilon
        self.momentum_coeff = momentum_coeff
        self.cache_coeff = cache_coeff

    def update(self, layer):

        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Momentum, used to accelerate SGD in the relevant direction
        layer.weight_momentums = \
            self.momentum_coeff * layer.weight_momentums + \
            (1 - self.momentum_coeff) * layer.dweights
        layer.bias_momentums = \
            self.momentum_coeff * layer.bias_momentums + \
            (1 - self.momentum_coeff) * layer.dbiases

        # Corrected momentum, due to the fact that we start at 0
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.momentum_coeff ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.momentum_coeff ** (self.iterations + 1))

        # Cache, used to reduce the learning rate over time
        layer.weight_cache = self.cache_coeff * layer.weight_cache + \
            (1 - self.cache_coeff) * layer.dweights ** 2
        layer.bias_cache = self.cache_coeff * layer.bias_cache + \
            (1 - self.cache_coeff) * layer.dbiases ** 2

        # Corrected cache, due to the fact that we start at 0
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.cache_coeff ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.cache_coeff ** (self.iterations + 1))

        # Update weights and biases
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)

    def update_learning_rate(self):
        if self.decay is not None:
            self.current_learning_rate = \
                self.learning_rate * (1. / (1. + self.decay * self.iterations))
            self.iterations += 1
