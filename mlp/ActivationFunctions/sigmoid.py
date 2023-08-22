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
