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

from utils.datasets import (
    load_dataset,
    print_dataset_info,
    split_dataset,
    save_dataset
)
from utils.plot import plot_histograms
from utils.column_names import (get_column_names, append_column_names)


INPUT_PATH = "../datasets/data.csv"
OUTPUT_PATH = "../datasets"
SPLIT_RATIO = 0.80


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
    plot_histograms(dataset, columns)

    # Split the dataset.
    train, validation = split_dataset(dataset, SPLIT_RATIO)

    # Save the train and the validation datasets.
    save_dataset(train, f"{OUTPUT_PATH}/train.csv")
    save_dataset(validation, f"{OUTPUT_PATH}/validation.csv")
