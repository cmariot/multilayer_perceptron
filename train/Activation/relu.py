import numpy as np


class ReluActivation:
    """
    Rectified Linear Unit
    Simpler and faster than sigmoid,
    speed, efficiency and non-linearity
    """

    def forward(self, layer_output):
        self.output = np.maximum(0, layer_output)
        return self.output


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    relu = ReluActivation()

    x = np.linspace(-10, 10)
    y = relu.forward(x)

    plt.title("Relu Activation Function")
    plt.plot(x, y)
    plt.show()
