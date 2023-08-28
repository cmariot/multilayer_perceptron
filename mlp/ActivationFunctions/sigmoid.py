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

from numpy import exp


class Sigmoid_Activation:

    def forward(self, input):
        """
        Sigmoid activation function
        Each input is transformed into a value between 0 and 1
        """
        self.input = input
        self.output = 1 / (1 + exp(-input))
        return self.output

    def backward(self, dvalues):
        """
        Derivative of the sigmoid function
        """
        self.dinput = dvalues * (1 - self.output) * self.output
        return self.dinput
