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
        Return an output in range 0 (for negative values) to 1
        (for positive values)
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


if __name__ == "__main__":

    # Plot the sigmoid function and its derivative on the same graph
    # between -10 and 10

    import matplotlib.pyplot as plt

    # Create 1000 equally spaced points between -10 and 10
    x = np.linspace(-10, 10, 1000)

    sigmoid = Sigmoid_Activation()

    # Plot the sigmoid function
    plt.plot(x, sigmoid.forward(x), label="Sigmoid")

    # Plot the derivative of the sigmoid function
    plt.plot(x, sigmoid.backward(), label="Sigmoid derivative")

    # Add a legend
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()
