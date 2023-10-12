# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    plot.py                                           :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/10/11 15:50:11 by cmariot          #+#    #+#              #
#    Updated: 2023/10/11 15:55:34 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import matplotlib.pyplot as plt
import numpy as np
from multilayer_perceptron.Metrics.confusion_matrix import confusion_matrix_


def plot_confusion_matrix(y, y_hat):
    try:
        # Plot the confusion matrix on the test set
        print(
            "\033[94m" +
            "\nConfusion matrix on the test set:\n" +
            "\033[0m"
        )
        plot_cm(
            confusion_matrix_(y, y_hat, df_option=True),
            classes=["Malignant", "Benign"],
            title="Confusion matrix on the test set"
        )
    except Exception as error:
        print(error)
        exit()


def plot_cm(cm, classes, title, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    # Put the values inside the confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Put the values inside the confusion matrix
            value = cm.iloc[i, j]
            plt.text(
                j,
                i,
                format(cm.iloc[i, j], "d"),
                ha="center",
                va="center",
                color="white" if value > 100 else "black"
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
