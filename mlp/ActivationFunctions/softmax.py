import numpy as np


class Softmax_Activation:

    def forward(self, input):
        self.input = input
        exp_values = \
            np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = \
            exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinput = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinput[index] = np.dot(jacobian_matrix,
                                        single_dvalues)
        return self.dinput
