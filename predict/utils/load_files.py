# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    load_files.py                                     :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/10/11 15:50:49 by cmariot          #+#    #+#              #
#    Updated: 2023/10/12 12:50:36 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import pickle
import pandas
import numpy as np


def load_model(path: str) -> object:
    try:
        with open(path, "rb") as file:
            model = pickle.load(file)
            return model
    except Exception as error:
        print(error)
        exit()


def normalize(x, x_min, x_max):
    try:
        if x.shape[1] != x_min.shape[0] or x.shape[1] != x_max.shape[0]:
            print("Error: The number of features in the dataset is not " +
                  "equal to the number of features in the model.")
            exit()
        x_norm = (x - x_min) / (x_max - x_min)
        x_norm = x_norm.T.to_numpy()
        return x_norm
    except Exception as error:
        print(error)
        exit()


def load_dataset(path: str, x_min, x_max) -> object:
    try:
        dataset = pandas.read_csv(path)
        if "Diagnosis" not in dataset.columns:
            x = dataset
            y = None
        else:
            x = dataset.drop("Diagnosis", axis=1)
            y = dataset["Diagnosis"]
            y = np.where(y == "M", 0, 1)
        x_norm = normalize(x, x_min, x_max)
        return x, x_norm, y

    except Exception as error:
        print(error)
        exit()


def save_predictions(y_hat, path):
    # Save the predictions in a csv file
    try:
        y_pred = np.where(y_hat == 0, "M", "B")
        y_pred = pandas.DataFrame(y_pred, columns=["Diagnosis"])
        y_pred.to_csv(path, index=False)
        print(
            "\033[94m" +
            f"\nPredictions saved in the {path} file." +
            "\033[0m")
    except Exception as error:
        print(error)
        exit()
