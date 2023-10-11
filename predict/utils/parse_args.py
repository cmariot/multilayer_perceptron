# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    parse_args.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/11 15:51:22 by cmariot           #+#    #+#              #
#    Updated: 2023/10/11 15:51:33 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import argparse


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