from numpy import exp


class Sigmoid_Activation:

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + exp(-input))

    # Backward pass
    def backward(self, dvalues):
        self.dinput = dvalues * (1 - self.output) * self.output
