# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    confusion_matrix.py                               :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:40:02 by cmariot          #+#    #+#              #
#    Updated: 2023/10/03 08:52:45 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import pandas
import numpy as np


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
        Compute confusion matrix to evaluate the accuracy of a classification.
        Args:
            y: a numpy.array for the correct labels
            y_hat: a numpy.array for the predicted labels
            labels: optional, a list of labels to index the matrix.
                    This may be used to reorder or select a subset of labels.
                    (default=None)
            df_option: optional, if set to True the function will return a
                       pandas DataFrame instead of a numpy array.
                       (default=False)
        Return:
            The confusion matrix as a numpy array or a pandas DataFrame
            according to df_option value.
            None if any error.
        Raises:
            This function should not raise any Exception.
    """

    try:
        if not isinstance(y_true, np.ndarray) \
                or not isinstance(y_hat, np.ndarray):
            print("Not a numpy array")
            return None
        y_true = y_true.argmax(axis=1)
        if y_true.shape != y_hat.shape:
            print("Shape error")
            print("y_true shape:", y_true.shape)
            print("y_hat shape:", y_hat.shape)
            return None
        if y_true.size == 0 or y_hat.size == 0:
            print("Empty array")
            return None
        y_hat = np.where(y_hat == 0, "Malignant", "Benign")
        y_true = np.where(y_true == 0, "Malignant", "Benign")
        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_hat)))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for i in range(len(labels)):
            for j in range(len(labels)):
                cm[i, j] = np.where((y_true == labels[i])
                                    & (y_hat == labels[j]))[0].shape[0]
        if df_option:
            cm = pandas.DataFrame(cm, index=labels, columns=labels)
        print(cm, "\n")
        return cm

    except Exception as error:
        print("Error:", error)
        return None
