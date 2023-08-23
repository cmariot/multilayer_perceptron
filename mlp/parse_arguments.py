import argparse


def parse_args():
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--layers",
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
            "--epochs",
            type=int,
            help="Number of epochs",
            default=84
        )

        parser.add_argument(
            "--loss",
            type=str,
            help="Loss function",
            default="binaryCrossentropy"
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            help="Batch size",
            default=32
        )

        parser.add_argument(
            "--learning_rate",
            type=float,
            help="Learning rate",
            default=10e-2
        )

        args = parser.parse_args()

        return (
            args.layers,
            args.activations,
            args.epochs,
            args.loss,
            args.batch_size,
            args.learning_rate
        )

    except Exception as e:
        print(e)
        exit()
