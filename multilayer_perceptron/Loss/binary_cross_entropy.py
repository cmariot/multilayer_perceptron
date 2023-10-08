# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    binary_cross_entropy.py                           :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:17:17 by cmariot          #+#    #+#              #
#    Updated: 2023/10/05 21:20:50 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np


class BinaryCrossEntropy_Loss:
    """
    The binary cross entropy loss is used for binary classification.
    The output of the model is a probability between 0 and 1.
    The target is 0 or 1.
    """

    # Compute the model error
    def calculate(self, output, y):

        try:

            # Clip the output to avoid log(0) and log(1 - 1) errors
            output_clipped = np.clip(output, 1e-10, 1 - 1e-10)

            # Calculate sample-wise loss ie for each sample of the batch
            # If y = 1, the loss is -log(output)
            # If y = 0, the loss is -log(1 - output)
            sample_loss = -(
                y * np.log(output_clipped)
                + (1 - y) * np.log(1 - output_clipped)
            )

            # Calculate the mean loss of the batch
            loss = np.mean(sample_loss)
            return loss

        except Exception as error:
            print("Error: can't calculate the Binary Cross Entropy loss")
            print(error)
            exit()

    # Used during the training phase
    def backward(self, output, y):

        try:

            # Number of samples in a batch (or in the dataset) and number of
            # labels
            n_samples, n_labels = output.shape

            # Clip the output to avoid division by 0
            output_clipped = np.clip(output, 1e-10, 1 - 1e-10)

            # Calculate and return the gradient
            gradient = -(
                y / output_clipped -
                (1 - y) / (1 - output_clipped)
            )
            return gradient / (n_samples * n_labels)

        except Exception:
            print("Error: can't calculate the Binary Cross Entropy gradient")
            exit()
