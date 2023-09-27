# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    sgd.py                                            :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:18:45 by cmariot          #+#    #+#              #
#    Updated: 2023/09/27 11:18:46 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

class StandardGradientDescent:

    def __init__(self, learning_rate, decay, momentum):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                    (1.0 / (1.0 + self.decay * self.iterations))

    def update(self, layer):

        if self.momentum:

            weight_updates = self.momentum * layer.weight_momentums - \
                    self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:

            weight_updates = - self.current_learning_rate * layer.dweights
            bias_updates = - self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1
