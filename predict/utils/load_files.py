# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    load_files.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/11 15:50:49 by cmariot           #+#    #+#              #
#    Updated: 2023/10/11 15:51:06 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

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


def load_dataset(path: str) -> object:
    try:
        dataset = pandas.read_csv(path)
        x = dataset.drop("Diagnosis", axis=1)
        y = dataset["Diagnosis"]
        y = np.where(y == "M", 0, 1)
        return x, y

    except Exception as error:
        print(error)
        exit()