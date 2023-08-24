# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    recall.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:52 by cmariot           #+#    #+#              #
#    Updated: 2023/08/24 14:39:53 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Recall tells you how much you can trust that your
    model is able to recognize ALL Class A objects.
    It is the percentage of all A objects that were properly
    classified by the model as Class A.
    You use recall when you want to control for False negatives.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report
                the precision_score (default=1)
    Return:
        The recall score as a float.
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
        fn = np.sum(np.logical_and(y == pos_label, y_hat != pos_label))

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    except Exception:
        return None
