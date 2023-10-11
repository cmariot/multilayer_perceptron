# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    plots.py                                          :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/10/03 08:35:45 by cmariot          #+#    #+#              #
#    Updated: 2023/10/03 08:36:56 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import pandas
import matplotlib.pyplot as plt


def print_metrics(training_metrics, validation_metrics):
    """
    Print the metrics computed on the training and validation datasets.
    """

    try:
        metrics_dataframe = {
            "Training": [
                training_metrics["loss"][-1],
                training_metrics["accuracy"][-1],
                training_metrics["recall"][-1],
                training_metrics["precision"][-1],
                training_metrics["f1_score"][-1]
            ],
            "Validation": [
                validation_metrics["loss"][-1],
                validation_metrics["accuracy"][-1],
                validation_metrics["recall"][-1],
                validation_metrics["precision"][-1],
                validation_metrics["f1_score"][-1]
            ]
        }
        print("\n",
              pandas.DataFrame(
                  metrics_dataframe,
                  index=[
                      "Loss",
                      "Accuracy",
                      "Recall",
                      "Precision",
                      "F1 score"
                  ]).to_string()
              )
    except Exception as e:
        print(e)
        exit()


def plot_loss(training_metrics, validation_metrics):
    try:
        plt.title("Loss evolution computed on training and validation sets")
        plt.plot(training_metrics["loss"], label="Training loss")
        plt.plot(validation_metrics["loss"], label="Validation loss")
        plt.legend()
        plt.show()
    except Exception as e:
        print(e)
        exit()


def plot_metrics(training_metrics, validation_metrics):
    try:
        # Create a figure with 4 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 9))
        descriptions = {
            "accuracy": "Accuracy (correctly classified / number of instances)",
            "precision": "Precision (increase when the number of false positives decreases)",
            "recall": "Recall (increases when the number of false negatives decreases)",
            "f1_score": "F1 score (harmonic mean of precision and recall))"
        }
        for metric, ax in zip(training_metrics, axs.flat):
            ax.set_title(descriptions[metric])
            ax.plot(training_metrics[metric], label="Training")
            ax.plot(validation_metrics[metric], label="Validation")
            ax.legend()
            ax.set_ylim([0, 1.1])
        plt.legend()
        plt.show()
    except Exception as e:
        print(e)
        exit()


def plot_loss_and_metrics(training_metrics, validation_metrics):
    """
    Plot the loss and the metrics computed on the training and validation sets.
    The loss and metrics are plotted on the same graph, but because of the
    different scales, the loss is plotted on the left y-axis and the metrics
    are plotted on the right y-axis.
    """
    try:
        fig = plt.figure(figsize=(15, 9))
        fig.suptitle("Loss and metrics evolution computed during training")

        # Plot the metrics on the right y-axis
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel("Y Metrics")

        colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
        for metric, color in zip(training_metrics, colors):
            ax1.plot(
                training_metrics[metric],
                label=f"Training {metric}",
                color=color
            )
            ax1.plot(
                validation_metrics[metric],
                label=f"Validation {metric}",
                color=color,
                linestyle="dotted"
            )

        ax1.set_ylim([0, 1.1])

        ax2 = ax1.twinx()

        # Plot the loss on the left y-axis
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Y Loss", color="tab:blue")
        ax2.plot(
            training_metrics["loss"],
            label="Training loss",
            color="tab:blue"
        )
        ax2.plot(
            validation_metrics["loss"],
            label="Validation loss",
            color="tab:blue",
            linestyle="dotted"
        )
        for label in ax2.get_yticklabels():
            label.set_color("tab:blue")

        # Legend of ax1 is center right, a little bit on the left
        ax1.legend(loc="center right")
        ax2.legend(loc="center right", bbox_to_anchor=(0.80, 0.5))

        plt.show()

    except Exception as e:
        print(e)
        exit()
