# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    parse_arguments.py                                :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:26 by cmariot          #+#    #+#              #
#    Updated: 2023/10/03 09:45:34 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import argparse


def parse_arguments():
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--train_path",
            type=str,
            help="Path to the train dataset",
            default="../datasets/train.csv"
        )

        parser.add_argument(
            "--validation_path",
            type=str,
            help="Path to the validation dataset",
            default="../datasets/validation.csv"
        )

        parser.add_argument(
            "--n_neurons",
            type=int,
            nargs="+",
            help="Number of neurons in each layer",
            default=[30, 24, 24, 2]
        )

        parser.add_argument(
            "--activations",
            type=str,
            nargs="+",
            help="Activation function in each layer",
            default=["sigmoid", "sigmoid", "sigmoid", "softmax"]
        )

        parser.add_argument(
            "--loss",
            type=str,
            help="Name of the loss function to use",
            default="binaryCrossentropy"
        )

        parser.add_argument(
            "--optimizer",
            type=str,
            help="Name of the optimizer function to use",
            default="sgd"
        )

        parser.add_argument(
            "--epochs",
            type=int,
            help="Number of epochs to train the model",
            default=500
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            help="Size of the batch used to train the model",
            default=32
        )

        parser.add_argument(
            "--learning_rate",
            type=float,
            help="Initial learning rate of the model",
            default=0.075
        )

        args = parser.parse_args()

        return (
            args.train_path,
            args.validation_path,
            args.n_neurons,
            args.activations,
            args.loss,
            args.optimizer,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )

    except Exception as error:
        print(error)
        exit()
