# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    parse_arguments.py                                :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:26 by cmariot          #+#    #+#              #
#    Updated: 2023/08/24 14:39:27 by cmariot         ###   ########.fr        #
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
            default=[30, 24, 24, 24, 2]
        )

        parser.add_argument(
            "--activations",
            type=str,
            nargs="+",
            help="Activation function in each layer",
            default=["sigmoid", "sigmoid", "sigmoid", "sigmoid", "softmax"]
        )

        parser.add_argument(
            "--loss",
            type=str,
            help="Name of the loss function to use",
            default="binaryCrossentropy"
        )

        parser.add_argument(
            "--epochs",
            type=int,
            help="Number of epochs to train the model",
            default=80
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            help="Size of the batch used to train the model",
            default=8
        )

        parser.add_argument(
            "--learning_rate",
            type=float,
            help="Initial learning rate of the model",
            default=0.005
        )

        parser.add_argument(
            "--decay",
            type=float,
            help="Decay of the learning rate, used to reduce it over time",
            default=0.01
        )

        parser.add_argument(
            "--momentum",
            type=float,
            help="Momentum of the model, used to accelerate the learning" +
            " and avoid local minima",
            default=0.007
        )

        args = parser.parse_args()

        return (
            args.train_path,
            args.validation_path,
            args.n_neurons,
            args.activations,
            args.loss,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.decay,
            args.momentum
        )

    except Exception as e:
        print(e)
        exit()
