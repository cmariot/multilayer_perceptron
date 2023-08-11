import numpy as np


class Relu_Activation:

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)

    # Backward pass
    def backward(self, dvalues):
        pass
