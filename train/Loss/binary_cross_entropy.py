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


class BinaryCrossEntropy_Loss:

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
        return np.mean(-(y_true * np.log(y_pred_clipped) +
                         (1 - y_true) * np.log(1 - y_pred_clipped)))

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
        y_pred_clipped = np.clip(dvalues, eps, 1 - eps)
        return -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped))
