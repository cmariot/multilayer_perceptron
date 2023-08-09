import numpy as np


class Loss:

    def calculate(self, y_true, y_pred):
        """
        Calculates the data and regularization losses
        given model output and ground truth values.
        """
        sample_losses = self.loss_elem(y_true, y_pred)
        loss = np.mean(sample_losses)
        return loss


class Categorical_Cross_Entropy(Loss):

    def loss_elem(self, y_pred, y_true, eps=1e-15):
        """
        Calculates the sample losses.
        """
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
        else:
            raise ValueError(
                "y_true should be 1D or 2D."
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
