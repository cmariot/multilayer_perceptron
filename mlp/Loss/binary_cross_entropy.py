import numpy as np


class BinaryCrossEntropy_Loss:

    def calculate(self, output, y):
        loss_elem = self.forward(output, y)
        loss = np.mean(loss_elem)
        return loss

    def forward(self, y_pred, y_true, eps=1e-15):
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -(y_true * np.log(y_pred_clipped) +
                 (1 - y_true) * np.log(1 - y_pred_clipped))
        loss_elem = np.mean(loss, axis=1)
        return loss_elem

    def backward(self, dvalues, y_true, eps=1e-15):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        y_pred_clipped = np.clip(dvalues, eps, 1 - eps)
        self.dinputs = -(y_true / y_pred_clipped -
                         (1 - y_true) / (1 - y_pred_clipped)) / outputs
        self.dinputs = self.dinputs / samples
