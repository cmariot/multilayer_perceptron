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
