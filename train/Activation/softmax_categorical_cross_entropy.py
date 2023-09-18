import numpy as np
from Activation.softmax import SoftmaxActivation
from Activation.categorical_cross_entropy import CategoricalCrossEntropy_Loss


class Softmax_Categorical_Cross_Entropy():

    def __init__(self):
        self.activation_function = SoftmaxActivation()
        self.loss_function = CategoricalCrossEntropy_Loss()

    def forward(self, inputs, y):
        self.activation_function.forward(inputs)
        self.output = self.activation_function.output
        return self.loss_function.calculate(self.output, y)
    
    def backward(self, dvalues, y):
        samples = len(dvalues)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y] -= 1
        self.dinputs = self.dinputs / samples
