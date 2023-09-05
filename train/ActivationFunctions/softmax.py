# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    softmax.py                                        :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:40:11 by cmariot          #+#    #+#              #
#    Updated: 2023/08/24 14:40:12 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np


class Softmax_Activation:

    def forward(self, input):
        exp = np.exp(input)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinput = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            self.dinput[index] = np.dot(jacobian_matrix,
                                        single_dvalues)
        return self.dinput


if __name__ == "__main__":

    # Plot the softmax function and its derivative on the same graph
    # between -10 and 10

    import matplotlib.pyplot as plt

    # Create 1000 equally spaced points between -10 and 10
    x = np.linspace(-10, 10, 1000).reshape(-1, 1)

    softmax = Softmax_Activation()

    # Plot the softmax function
    plt.plot(x, softmax.forward(x), label="Softmax")

    # Plot the derivative of the softmax function
    plt.plot(x, softmax.backward(x), label="Softmax derivative")

    # Add a legend
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()
