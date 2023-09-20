import numpy as np
from Activation.softmax import SoftmaxActivation
from Activation.binary_cross_entropy import BinaryCrossEntropy_Loss


class Softmax_Binary_Cross_Entropy():

    def __init__(self):
        self.activation_function = SoftmaxActivation()
        self.loss_function = BinaryCrossEntropy_Loss()

    def forward(self, inputs, y):
        self.activation_function.forward(inputs)
        self.output = self.activation_function.output
        return self.loss_function.calculate(self.output, y)

    def backward(self, dvalues, y):
        pass

