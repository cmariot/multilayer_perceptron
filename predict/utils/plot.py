# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/11 15:50:11 by cmariot           #+#    #+#              #
#    Updated: 2023/10/11 15:55:34 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure(figsize=(8, 7))
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
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
