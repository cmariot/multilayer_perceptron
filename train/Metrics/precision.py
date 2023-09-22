# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    precision.py                                      :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:55 by cmariot          #+#    #+#              #
#    Updated: 2023/08/24 14:39:56 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Precision tells you how much you can trust your
    model when it says that an object belongs to Class A.
    More precisely, it is the percentage of the objects
    assigned to Class A that really were A objects.
    You use precision when you want to control for False positives.
    Args:
        y: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report
                the precision_score (default=1)
    Return:
        The precision score as a float.
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

        tp = np.sum(np.logical_and(y == pos_label, y == y_hat))
        fp = np.sum(np.logical_and(y != pos_label, y_hat == pos_label))

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    except Exception:
        return 0