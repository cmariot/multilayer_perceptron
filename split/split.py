# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    split.py                                          :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/26 11:39:25 by cmariot          #+#    #+#              #
#    Updated: 2023/09/28 11:24:14 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import pandas
import matplotlib.pyplot as plt


INPUT_PATH = "../datasets/data.csv"
OUTPUT_PATH = "../datasets"
SPLIT_RATIO = 0.80


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


def get_column_names():
    """"
    The original dataset has no column names.
    The labels was found here:
    https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
    This function returns the list of the column names for this dataframe.
    """
    try:

        # The 30 features are computed from a digitized image of a fine needle
        # aspirate (FNA) of a breast mass.
        # List of the 10 features computed for each cell nucleus:
        features = [
            "Radius",            # mean of distances center / perimeter
            "Texture",           # standard deviation of gray-scale values
            "Perimeter",
            "Area",
            "Smoothness",        # local variation in radius lengths
            "Compactness",       # perimeter^2 / area - 1.0
            "Concavity",         # severity of concave portions of the contour
            "Concave points",    # number of concave portions of the contour
            "Symmetry",
            "Fractal dimension"  # "coastline approximation" - 1
        ]

        # The final feature list is composed of 30 features.
        # The 10 first features are the mean of the cells.
        # The 10 next features are the standard error of the cells.
        # The 10 last features are the worst (mean of the three largest values)
        # of the cells.
        mean = [f"{feature} mean" for feature in features]
        se = [f"{feature} SE" for feature in features]
        worst = [f"{feature} worst" for feature in features]

        # The first two columns are the ID number and the diagnosis.
        return ["ID number", "Diagnosis"] + mean + se + worst

    except Exception as error:
        fatal_error(error)


def append_column_names(dataset, columns):
    try:
        if len(dataset.columns) != len(columns):
            fatal_error("The dataset columns length is not equal to the",
                        "columns length.")
        # Add the column names.
        dataset.columns = columns
        # Return the dataset without the 'ID number' column.
        return dataset.drop("ID number", axis=1)
    except Exception as error:
        fatal_error(error)


def print_dataset_info(dataset):
    try:
        print("Dataset informations:")
        print(dataset.head(), "\n")
        print(dataset.describe(include="all"))
    except Exception as error:
        fatal_error(error)


def plot_histograms(dataset):
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


def fatal_error(error_message):
    print("Error:", error_message)
    exit()


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

    # Load the original dataset.
    dataset = load_dataset(INPUT_PATH)

    # Get the column names list.
    columns = get_column_names()

    # Add the column names.
    dataset = append_column_names(dataset, columns)

    # Print the dataset informations.
    print_dataset_info(dataset)

    # Histogram of the features.
    plot_histograms(dataset)

    # Split the dataset.
    train, validation = split_dataset(dataset, SPLIT_RATIO)

    # Save the train and the validation datasets.
    save_dataset(train, f"{OUTPUT_PATH}/train.csv")
    save_dataset(validation, f"{OUTPUT_PATH}/validation.csv")
