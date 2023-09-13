import numpy as np


class Loss:
    """
    Loss is a base class for all loss functions.
    """

    def calculate(self, y_hat, y):
        sample_loss = self.forward(y_hat, y)
        data_loss = np.mean(sample_loss)
        return data_loss

    def forward(self, y_hat, y):
        raise NotImplementedError


if __name__ == "__main__":

    loss = Loss()

    y = np.array([1, 0, 1, 1])
    y_hat = np.array([0.5, 0.5, 0.5, 0.5])

    try:
        print(loss.calculate(y_hat, y))
    except NotImplementedError:
        print("NotImplementedError")
