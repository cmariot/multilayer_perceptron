import numpy as np


class StepActivation:
    """
    On/Off activation function based on output of a neuron
    """

    def forward(self, layer_output):
        self.output = np.where(layer_output > 0, 1, 0)
        return self.output


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    step = StepActivation()

    x = np.linspace(-10, 10)
    y = step.forward(x)

    plt.title("Step Activation Function")
    plt.plot(x, y)
    plt.show()
