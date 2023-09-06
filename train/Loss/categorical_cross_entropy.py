from .loss import Loss
import numpy as np


class CategoricalCrossEntropy_Loss(Loss):

    def forward(self, output, y_true, eps=1e-15):
        """
        Compute the categorical cross entropy loss.
        Args:
            output (np.array): Output of the last layer.
            y_true (np.array): True values.
            eps (float): Epsilon to avoid division by zero.
        Returns:
            float: The categorical cross entropy loss.
        """
        samples = len(output)
        output_clipped = np.clip(output, eps, 1 - eps)
        if len(y_true.shape) == 1:
            correct_confidences = output_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(output_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, eps=1e-15):
        pass
