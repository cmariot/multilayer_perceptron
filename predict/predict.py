# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    predict.py                                        :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:21:00 by cmariot          #+#    #+#              #
#    Updated: 2023/09/27 11:27:08 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import argparse
import pickle
import pandas
import numpy as np

from multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron
from multilayer_perceptron.Metrics.accuracy import accuracy_score_


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

    # Normalize the features
    x_norm = (x - model.x_min) / (model.x_max - model.x_min)

    # Predict the target
    y_pred = model.forward(x_norm)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.where(y_pred == 0, "M", "B")
    y = y.to_numpy()

    accuracy = accuracy_score_(y, y_pred)
    print(f"Accuracy: {accuracy}")
