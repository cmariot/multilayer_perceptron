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
import numpy as np


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

        print(
            "\033[94m" +
            "\nFinal metrics on the training and validation sets :\n" +
            "\033[0m\n" +
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
            "accuracy":
                "Accuracy (correctly classified / number of instances)",
            "precision":
                "Precision (increase when nb of false positives decreases)",
            "recall":
                "Recall (increases when nb of false negatives decreases)",
            "f1_score":
                "F1 score (harmonic mean of precision and recall))"
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


def plot_3d(x_validation, x_validation_norm, y_validation, model):

    try:

        plot_features = [
            "Radius mean",
            "Texture mean",
            "Smoothness mean"
        ]

        y_hat = model.predict(x_validation_norm)
        y_validation = np.argmax(y_validation, axis=0).reshape(-1, 1)

        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(projection='3d')

        colors = {
            "true positive": "green",
            "true negative": "blue",
            "false positive": "red",
            "false negative": "orange"
        }

        validation_colors = []
        labels = []
        for i in range(len(y_validation)):
            if y_validation[i] == 0 and y_hat[i] == 0:
                validation_colors.append(colors["true positive"])
                labels.append("True positive")
            elif y_validation[i] == 1 and y_hat[i] == 1:
                validation_colors.append(colors["true negative"])
                labels.append("True negative")
            elif y_validation[i] == 0 and y_hat[i] == 1:
                validation_colors.append(colors["false negative"])
                labels.append("False negative")
            elif y_validation[i] == 1 and y_hat[i] == 0:
                validation_colors.append(colors["false positive"])
                labels.append("False positive")

        ax.scatter(
            x_validation[plot_features[0]],
            x_validation[plot_features[1]],
            x_validation[plot_features[2]],
            c=validation_colors,
        )

        # Title
        ax.set_title(
            "Tridimensional Insight: Mapping cellular target through " +
            "three features with colorful predictions"
        )

        # Axis labels
        ax.set_xlabel(plot_features[0])
        ax.set_ylabel(plot_features[1])
        ax.set_zlabel(plot_features[2])

        # Add a legend
        for key, value in colors.items():
            ax.scatter([], [], [], c=value, label=key)
        ax.legend()

        plt.show()

    except Exception as e:
        print(e)
        exit()
