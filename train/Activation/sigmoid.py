import numpy as np


class SigmoidActivation:
    """
    Returns a value between 0 and 1
    Add non-linearity to the network
    """

    def forward(self, layer_output):
        self.output = 1 / (1 + np.exp(-layer_output))
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
        return self.dinputs
