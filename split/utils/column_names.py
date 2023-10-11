# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    column_names.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/11 15:46:33 by cmariot           #+#    #+#              #
#    Updated: 2023/10/11 16:00:07 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from utils.fatal_error import fatal_error


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
