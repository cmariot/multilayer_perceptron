import numpy as np


class ReluActivation:
    """
    Rectified Linear Unit
    Simpler and faster than sigmoid,
    speed, efficiency and non-linearity
    """

    def forward(self, layer_output):
        self.input = layer_output
        self.output = np.maximum(0, layer_output)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
        return self.dinputs    


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    relu = ReluActivation()

    x = np.linspace(-10, 10)
    y = relu.forward(x)

    plt.title("Relu Activation Function")
    plt.plot(x, y)
    plt.show()
