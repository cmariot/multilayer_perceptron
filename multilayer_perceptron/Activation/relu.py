# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    relu.py                                           :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:17:31 by cmariot          #+#    #+#              #
#    Updated: 2023/09/27 11:17:33 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np


class ReluActivation:
    """
    Rectified Linear Unit
    Simpler and faster than sigmoid,
    speed, efficiency and non-linearity
    """

    def forward(self, layer_output):
        self.input = layer_output
        self.output = np.maximum(0, layer_output)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
        return self.dinputs
