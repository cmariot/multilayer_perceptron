import numpy as np


def loss_selection(loss: str):

    losses = {
        "binaryCrossentropy": BinaryCrossentropy
    }
    return losses[loss]() if loss in losses else None


class Loss:

    def calculate(self, y_true, y_pred):
        """
        Calculates the data and regularization losses
        given model output and ground truth values.
        """
        loss_elem = self.forward(y_true, y_pred)
        loss = np.mean(loss_elem)
        return loss


class BinaryCrossentropy(Loss):

    def __init__(self):
        print("BinaryCrossentropy")

    def forward(self, y_true, y_pred, eps=1e-15):
        """
        Calculates the sample losses.
        """
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -(y_true * np.log(y_pred_clipped) +
                 (1 - y_true) * np.log(1 - y_pred_clipped))
        return loss

    def backward(self, dvalues, y_true, eps=1e-15):
        samples = len(dvalues)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(dvalues, eps, 1 - eps)
        # Calculate gradient
        self.dinputs = -(y_true / y_pred_clipped -
                         (1 - y_true) / (1 - y_pred_clipped)) / samples
