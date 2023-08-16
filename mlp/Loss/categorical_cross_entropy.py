import numpy as np


class CategoricalCrossEntropy_Loss:

    def calculate(self, output, y):
        loss_elem = self.forward(output, y)
        loss = np.mean(loss_elem)
        return loss

    def forward(self, y_pred, y_true, eps=1e-15):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        correct_confidences = np.clip(correct_confidences, eps, 1 - eps)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true, eps=1e-15):
        pass
