# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    get_datasets.py                                   :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 19:51:57 by cmariot          #+#    #+#              #
#    Updated: 2023/08/24 19:52:01 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import pandas
import numpy as np


def load_dataset(path: str) -> pandas.DataFrame:
    try:
        dataset = pandas.read_csv(path)
        return dataset
    except Exception:
        print(f"Error: can't load the dataset {path}")
        exit()


def get_training_data(dataset_path: str) -> tuple:
    """
    Load the training dataset, normalize the features and
    replace the target labels by 0/1.
    """

    # Load the training dataset
    train_data = load_dataset(dataset_path)

    def normalize_train(x_train: pandas.DataFrame) -> tuple:
        """
        Normalize the features of the training dataset.
        All the features will be between 0 and 1.
        Return the normalized features, the minimum and the maximum
        values of each feature, used to normalize the validation set.
        """
        x_min = x_train.min()
        x_max = x_train.max()
        x_train_norm = (x_train - x_min) / (x_max - x_min)
        return (
            x_train_norm.to_numpy(),
            x_min,
            x_max
        )

    try:

        # Get the features and normalize them
        x_train = train_data.drop("Diagnosis", axis=1)
        x_train_norm, x_min, x_max = normalize_train(x_train)

        # Get the target and replace the labels by 0/1
        y_train = train_data["Diagnosis"]
        y_train = np.where(y_train == "M", 0, 1)
        y_train = y_train.reshape(-1, 1)
        y_train = np.array(
            [
                [1, 0] if y == 0 else [0, 1]
                for y in y_train
            ]
        )
        return (
            x_train_norm,
            y_train,
            x_min,
            x_max
        )

    except Exception:
        print("Error: can't get the training data.")
        exit()


def get_validation_data(
        dataset_path: str,
        x_min: pandas.Series,
        x_max: pandas.Series) -> tuple:
    """
    Load the validation dataset, normalize the features and
    replace the target labels by 0/1.
    """

    # Load the validation dataset
    validation_data = load_dataset(dataset_path)

    def normalize_validation(x_validation: pandas.DataFrame,
                             x_min: pandas.Series,
                             x_max: pandas.Series) -> pandas.DataFrame:
        """
        Normalize the features of the validation dataset.
        Use the minimum and the maximum values of each feature
        of the training set to normalize the validation set.
        All the features will be between 0 and 1.
        """
        x_validation_norm = (x_validation - x_min) / (x_max - x_min)
        return x_validation_norm.to_numpy()

    try:

        # Get the features and normalize them
        x_validation = validation_data.drop("Diagnosis", axis=1)
        x_validation_norm = normalize_validation(x_validation, x_min, x_max)

        # Get the target and replace the labels by 0/1
        y_validation = validation_data["Diagnosis"]
        y_validation = np.where(y_validation == "M", 0, 1)
        y_validation = y_validation.reshape(-1, 1)
        y_validation = np.array(
            [
                [1, 0] if y == 0 else [0, 1]
                for y in y_validation
            ]
        )
        return x_validation_norm, y_validation

    except Exception:
        print("Error: can't get the validation data.")
        exit()
