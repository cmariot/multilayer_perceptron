# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    metrics.py                                        :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:43 by cmariot          #+#    #+#              #
#    Updated: 2023/08/24 14:39:44 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import matplotlib.pyplot as plt


def plot_metrics(training_metrics: dict, validation_metrics: dict):
    """
    Plot the metrics evolution.
    """

    try:

        fig, ax = plt.subplots(2, 2, figsize=(15, 8))

        fig.suptitle(
            "Metrics evolution, computed on the training and " +
            "validation sets during the model training."
        )

        for i in range(2):

            for j in range(2):

                metric_name = list(training_metrics.keys())[i * 2 + j]
                validation_values = validation_metrics[metric_name]
                training_values = training_metrics[metric_name]

                ax[i, j].plot(
                    training_values,
                    label=f"training {metric_name}",
                    color='b',
                    linestyle=':'
                )
                ax[i, j].plot(
                    validation_values,
                    label=f"validation {metric_name}",
                    color='b'
                )

                # Display the last value of the metric with the text method
                ax[i, j].text(
                    len(validation_values) - 1,
                    validation_values[-1],
                    f" {validation_values[-1]:.4f}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    color="b"
                )

                ax[i, j].set_xlabel("Epochs")
                ax[i, j].set_xlim(0, len(validation_values) + 8)
                ax[i, j].set_ylabel(metric_name)
                ax[i, j].set_ylim(-0.05, 1.05)
                ax[i, j].grid()
                ax[i, j].legend()

        plt.show()

    except Exception as e:

        print(f"Error while plotting the loss: {e}")
        exit()
