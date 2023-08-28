# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    sigmoid.py                                        :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:40:14 by cmariot          #+#    #+#              #
#    Updated: 2023/08/24 14:40:15 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np

class Sigmoid_Activation:

    def forward(self, input):
        """
        Sigmoid activation function
        Return an output in range 0 (for negative values) to 1 (for positive values
        It adds non-linearity to the network
        """
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, dvalues):
        """
        Derivative of the sigmoid function
        """
        self.dinputs = dvalues * (1 - self.output) * self.output
        return self.dinputs
