# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    f1_score.py                                       :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:58 by cmariot          #+#    #+#              #
#    Updated: 2023/09/27 11:14:53 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np
from multilayer_perceptron.Metrics.precision import precision_score_
from multilayer_perceptron.Metrics.recall import recall_score_


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    F1 score combines precision and recall in one single measure.
    You use the F1 score when want to control both
    False positives and False negatives.
    Args:
        y: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report
                the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if not isinstance(y, np.ndarray) \
                or not isinstance(y_hat, np.ndarray):
            return None

        if y.shape != y_hat.shape:
            return None

        if y.size == 0 or y_hat.size == 0:
            return None

        if not isinstance(pos_label, (int, str)):
            return None

        precision = precision_score_(y, y_hat, pos_label)
        recall = recall_score_(y, y_hat, pos_label)

        if precision is None or recall is None:
            return 0.0
        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    except Exception:
        return 0.0
