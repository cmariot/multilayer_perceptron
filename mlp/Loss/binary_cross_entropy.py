import numpy as np


class BinaryCrossEntropy_Loss:

    def forward(self, y_hat, y, eps=1e-15):
        m = y_hat.shape[0]
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        const = -1.0
        dot1 = np.dot(y.T, np.log(y_hat_clipped))
        dot2 = np.dot((1 - y).T, np.log(1 - y_hat_clipped))
        return np.mean(const * (dot1 + dot2)) 
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -(y_true * np.log(y_pred_clipped) +
                 (1 - y_true) * np.log(1 - y_pred_clipped))
        loss_elem = np.mean(loss, axis=-1)
        self.output = np.mean(loss_elem)
        return self.output

    def gradient(self, dvalues, y_true, eps=1e-15):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        y_pred_clipped = np.clip(dvalues, eps, 1 - eps)
        self.dinputs = -(y_true / y_pred_clipped -
                         (1 - y_true) / (1 - y_pred_clipped)) / outputs
        self.dinputs = self.dinputs / samples
        return self.dinputs
