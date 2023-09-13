from Loss.loss import Loss
import numpy as np


class CategoricalCrossEntropy_Loss(Loss):

    def forward(self, y_hat, y):
        samples = len(y_hat)
        y_hat_clipped = np.clip(y_hat, 1e-7, 1 - 1e-7)
        if len(y.shape) == 1:
            correct_confidences = y_hat_clipped[range(samples), y]
        elif len(y.shape) == 2:
            correct_confidences = np.sum(y_hat_clipped * y, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


if __name__ == "__main__":

    softmax_output = np.array([[0.7, 0.1, 0.2],
                               [0.1, 0.5, 0.4],
                               [0.02, 0.9, 0.08]])

    class_targets = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 1, 0]])

    loss = CategoricalCrossEntropy_Loss()

    print(loss.calculate(softmax_output, class_targets))
