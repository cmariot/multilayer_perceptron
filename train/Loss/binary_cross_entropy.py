# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    binary_cross_entropy.py                           :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:40:08 by cmariot          #+#    #+#              #
#    Updated: 2023/08/24 20:06:08 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np
from .loss import Loss


class BinaryCrossEntropy_Loss(Loss):

    def forward(self, y_pred, y_true, eps=1e-15):
        """
        Compute the binary cross entropy loss.
        Args:
            y_pred (np.array): Predictions.
            y_true (np.array): True values.
            eps (float): Epsilon to avoid division by zero.
        Returns:
            float: The binary cross entropy loss.
        """
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        sample_loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_loss = np.mean(sample_loss, axis=-1)
        return sample_loss

    def gradient(self, dvalues, y_true, eps=1e-15):
        """
        Compute the gradient of the loss function.
        Args:
            dvalues (np.array): The derivative of the activation function.
            y_true (np.array): True values.
            eps (float): Epsilon to avoid division by zero.
        Returns:
            np.array: The gradient of the loss function.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, eps, 1 - eps)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples
        return self.dinputs
