# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    predict.py                                        :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:21:00 by cmariot          #+#    #+#              #
#    Updated: 2023/10/05 21:14:50 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import argparse
import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt

from multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron
from multilayer_perceptron.Metrics.accuracy import accuracy_score_
from multilayer_perceptron.Metrics.precision import precision_score_
from multilayer_perceptron.Metrics.recall import recall_score_
from multilayer_perceptron.Metrics.f1_score import f1_score_
from multilayer_perceptron.Metrics.confusion_matrix import confusion_matrix_


def parse_arguments():

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--predict_path",
            type=str,
            help="Path to the train dataset",
            default="../datasets/validation.csv"
        )

        parser.add_argument(
            "--model_path",
            type=str,
            help="Path to the trained Multilayer Perceptron model",
            default="../model.pkl"
        )

        args = parser.parse_args()

        return (
            args.predict_path,
            args.model_path
        )

    except Exception as error:
        print(error)
        exit()


def load_model(path: str) -> object:
    try:
        with open(path, "rb") as file:
            model = pickle.load(file)
            return model
    except Exception as error:
        print(error)
        exit()


def load_dataset(path: str) -> object:
    try:
        dataset = pandas.read_csv(path)
        x = dataset.drop("Diagnosis", axis=1)
        y = dataset["Diagnosis"]
        return x, y

    except Exception as error:
        print(error)
        exit()


def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure()
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


if __name__ == "__main__":

    # Parse command line arguments
    (
        predict_path,  # Path to the prediction dataset
        model_path     # Path to the model
    ) = parse_arguments()

    # Load the model
    model: MultilayerPerceptron = load_model(model_path)

    # Load the dataset
    x, y = load_dataset(predict_path)
    # Convert the targets to a numpy array
    y = np.where(y == "M", 0, 1)

    # Normalize the features of the dataset and convert it to a numpy array
    x_norm = (x - model.x_min) / (model.x_max - model.x_min)
    x_norm = x_norm.T.to_numpy()

    # Predict the dataset
    y_hat = model.predict(x_norm)

    df = pandas.DataFrame(
        {
            "Binary cross entropy loss": model.loss.calculate(y_hat, y),
            "Accuracy": accuracy_score_(y, y_hat),
            "Recall": recall_score_(y, y_hat),
            "Precision": precision_score_(y, y_hat),
            "F1": f1_score_(y, y_hat)
        },
        index=["Test set"]
    )
    print(df.T, "\n")

    # Plot the confusion matrix on the validation set
    plot_confusion_matrix(
        confusion_matrix_(y, y_hat, df_option=True),
        classes=["Malignant", "Benign"],
        title="Confusion matrix on the training set"
    )

    # Save the predictions in a csv file
    y_pred = np.where(y_hat == 0, "M", "B")
    y_pred = pandas.DataFrame(y_pred, columns=["Diagnosis"])
    y_pred.to_csv("../datasets/predictions.csv", index=False)
