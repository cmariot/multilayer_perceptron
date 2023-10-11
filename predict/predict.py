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

import pandas
import numpy as np

from utils.parse_args import parse_arguments
from utils.load_files import load_model, load_dataset
from utils.plot import plot_confusion_matrix

from multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron
from multilayer_perceptron.Metrics.accuracy import accuracy_score_
from multilayer_perceptron.Metrics.precision import precision_score_
from multilayer_perceptron.Metrics.recall import recall_score_
from multilayer_perceptron.Metrics.f1_score import f1_score_
from multilayer_perceptron.Metrics.confusion_matrix import confusion_matrix_


def header():
    print("""
              _ _   _     __
  /\\/\\  _   _| | |_(_)   / /  __ _ _   _  ___ _ __
 /    \\| | | | | __| |  / /  / _` | | | |/ _ \\ '__|
/ /\\/\\ \\ |_| | | |_| | / /__| (_| | |_| |  __/ |
\\/    \\/\\__,_|_|\\__|_| \\____/\\__,_|\\__, |\\___|_|
   ___                        _    |___/
  / _ \\___ _ __ ___ ___ _ __ | |_ _ __ ___  _ __
 / /_)/ _ \\ '__/ __/ _ \\ '_ \\| __| '__/ _ \\| '_ \\
/ ___/  __/ | | (_|  __/ |_) | |_| | | (_) | | | |
\\/    \\___|_|  \\___\\___| .__/ \\__|_|  \\___/|_| |_|
                       |_|
""")


if __name__ == "__main__":

    header()
    
    # Parse command line arguments
    (
        predict_path,  # Path to the prediction dataset
        model_path     # Path to the model
    ) = parse_arguments()

    # Load the model
    model: MultilayerPerceptron = load_model(model_path)

    # Load the dataset
    x, y = load_dataset(predict_path)

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
