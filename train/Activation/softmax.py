import numpy as np


class SoftmaxActivation:
    """
    Softmax activation function is usually used
    in the output layer of a classification model
    It returns a value between 0 and 1,
    1 being the highest probability
    """

    def forward(self, layer_output):
        exp_values = np.exp(
            layer_output -
            np.max(layer_output, axis=1, keepdims=True)
        )
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(
                zip(self.output, dvalues)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        return self.dinputs
