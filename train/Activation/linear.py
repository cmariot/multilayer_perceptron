import numpy as np


class LinearActivation:
    """
    Linear activation function is usually used
    in the output layer of a regression model
    y = x
    """

    def forward(self, layer_output):
        self.output = layer_output
        return self.output


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    linear = LinearActivation()

    x = np.linspace(-10, 10)
    y = linear.forward(x)

    plt.title("Linear Activation Function")
    plt.plot(x, y)
    plt.show()
