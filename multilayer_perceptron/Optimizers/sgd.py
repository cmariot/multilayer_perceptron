# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    sgd.py                                            :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:18:45 by cmariot          #+#    #+#              #
#    Updated: 2023/09/28 13:06:16 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

class StandardGradientDescent:

    def __init__(self,
                 learning_rate,
                 ):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases
