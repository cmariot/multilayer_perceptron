# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/11 15:46:43 by cmariot           #+#    #+#              #
#    Updated: 2023/10/11 15:46:44 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
from utils.fatal_error import fatal_error


def plot_histograms(dataset, columns):
    try:
        # Diffrence between the benign and malignant tumors.
        benign = dataset[dataset["Diagnosis"] == "B"]
        malignant = dataset[dataset["Diagnosis"] == "M"]
        # Plot the histograms.
        axes = plt.subplots(6, 5, figsize=(15, 9))
        plt.suptitle("Histogram of the features", fontsize=16)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        for i in range(6):
            for j in range(5):
                feature = columns[i * 5 + j + 2]
                axes[1][i][j].hist(benign[feature], bins=30, alpha=0.5,
                                   label="Benign", color="blue")
                axes[1][i][j].hist(malignant[feature], bins=30, alpha=0.5,
                                   label="Malignant", color="red")
                axes[1][i][j].set_title(feature, fontsize=10)
                axes[1][i][j].legend(loc="upper right", fontsize=8)
        plt.show()
    except Exception as error:
        fatal_error(error)