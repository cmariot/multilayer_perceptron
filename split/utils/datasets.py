# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    datasets.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/11 15:46:37 by cmariot           #+#    #+#              #
#    Updated: 2023/10/11 15:46:38 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from utils.fatal_error import fatal_error
import pandas


def load_dataset(path):
    """
    Load the dataset from a csv file path and return a pandas dataframe.
    """
    try:
        dataset = pandas.read_csv(path)
        print(f"Dataset {path} successfully loaded.",
              f"\nOriginal dataset contains {len(dataset)} samples.\n")
        return dataset
    except Exception as error:
        fatal_error(error)


def print_dataset_info(dataset):
    try:
        print("Dataset informations:")
        print(dataset.head(), "\n")
        print(dataset.describe(include="all"))
    except Exception as error:
        fatal_error(error)


def split_dataset(dataset, train_percentage):
    try:
        # Check the train_percentage value.
        if train_percentage >= 1 or train_percentage <= 0:
            fatal_error("train_percentage must be between 0 and 1.")

        # Shuffle the dataset.
        dataset = dataset.sample(frac=1)

        # Get the index of the train set.
        dataset_len = len(dataset)
        train_begin_index = 0
        train_end_index = int(dataset_len * train_percentage)
        validation_begin_index = train_end_index
        validation_end_index = dataset_len

        # Check the train and validation sets length.
        if train_end_index == 0 or validation_begin_index == dataset_len:
            fatal_error("train_percentage is too low or too high.")

        # Split the dataset.
        train = dataset[train_begin_index:train_end_index]
        validation = dataset[validation_begin_index:validation_end_index]

        print("\nDataset splitted into train and validation sets",
              f"with a {train_percentage} ratio :")
        print(f"- Train set: {len(train)} samples.")
        print(f"- Validation set: {len(validation)} samples.\n")

        return train, validation

    except Exception as error:
        fatal_error(error)


def save_dataset(dataset_df: pandas.DataFrame, new_path: str):
    try:
        dataset_df.to_csv(new_path, index=False)
        print(f"Dataset saved as {new_path}")
    except Exception as error:
        fatal_error(error)