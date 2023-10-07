# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    MultilayerPerceptron.py                           :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/26 14:41:05 by cmariot          #+#    #+#              #
#    Updated: 2023/10/05 21:19:43 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np
import pandas
import pickle

from multilayer_perceptron.layer import Layer

from multilayer_perceptron.Loss.binary_cross_entropy \
        import BinaryCrossEntropy_Loss

from multilayer_perceptron.Optimizers.sgd import StandardGradientDescent
from multilayer_perceptron.Optimizers.adagrad import AdaGrad

from multilayer_perceptron.ft_progress import ft_progress
from multilayer_perceptron.Metrics.accuracy import accuracy_score_
from multilayer_perceptron.Metrics.precision import precision_score_
from multilayer_perceptron.Metrics.recall import recall_score_
from multilayer_perceptron.Metrics.f1_score import f1_score_


class MultilayerPerceptron:

    # *********************************************************************** #
    # Multilayer Perceptron class is used to :                                #
    # - design a personalised neural network architecture                     #
    # - train the neural network with a training dataset                      #
    # - predict the output of a test dataset                                  #
    # *********************************************************************** #

    # *********************************************************************** #
    #                                                                         #
    # Multilayer Perceptron constructor :                                     #
    #                                                                         #
    # *********************************************************************** #

    def __init__(
        self,
        n_neurons: list,         # Number of neurons in each layer
        activations: list,       # Activation function in each layer
        loss_name: str,          # Loss function,
        optimizer_name: str,     # Optimizer function
        learning_rate: float,    # Initial learning rate
        decay: float,            # Learning rate decay
        momentum: float,         # Momentum
        train_set_shape: tuple,  # Shape of the training dataset
        epochs: int,             # Number of epochs
        batch_size: int,         # Batch size
        x_min: list,             # Min values of the features
        x_max: list,             # Max values of the features
    ):

        try:

            # *************************************************************** #
            # Layers list initialization                                      #
            # *************************************************************** #

            if len(n_neurons) != len(activations):
                raise Exception("Number of layers and activations mismatch")

            self.layers = []
            n_layers = len(n_neurons)
            n_features = train_set_shape[0]
            n_samples = train_set_shape[1]

            if n_layers < 2:
                raise Exception("The network must have at least 2 layers")
            elif n_samples < 1:
                raise Exception("The dataset must contain at least 1 sample")
            elif n_features < 1:
                raise Exception("The dataset must contain at least 1 feature")

            for i in range(n_layers):
                self.layers.append(
                    Layer(
                        n_neurons=n_neurons[i],
                        n_inputs=n_features if i == 0 else n_neurons[i - 1],
                        activation_function=activations[i]
                    )
                )

            # *************************************************************** #
            # Loss function initialization
            # *************************************************************** #

            available_losses = {
                "binaryCrossentropy": BinaryCrossEntropy_Loss,
                # Add more losses here
            }

            if loss_name in available_losses:
                self.loss = available_losses[loss_name]()
            else:
                raise Exception("Loss function unavailable")

            # *************************************************************** #
            # Optimizer initialization
            # *************************************************************** #

            if learning_rate < 0:
                raise Exception("The learning rate must be greater than 0")

            available_optimizers = {
                "sgd": StandardGradientDescent,
                "adagrad": AdaGrad,
                # Add more optimizers here
            }

            if optimizer_name in available_optimizers:
                self.optimizer = available_optimizers[optimizer_name](
                        learning_rate,
                        decay,
                        momentum
                )
            else:
                raise Exception("Optimizer unavailable")

            # *************************************************************** #
            # Training parameters initialization
            # *************************************************************** #

            self.epochs = epochs
            self.batch_size = batch_size
            self.n_samples = n_samples

            if batch_size < 1:
                raise Exception("The batch size must be greater than 0")
            elif epochs < 1:
                raise Exception("The number of epochs must be greater than 0")

            # The number of batches is the number of times the model will be
            # updated during one epoch
            self.n_batch = n_samples // batch_size
            if n_samples % batch_size != 0:
                self.n_batch += 1

            if self.n_batch < 1:
                raise Exception(
                    "The batch size must be smaller than the number of samples"
                )

            # Normalization parameters, used to normalize the test dataset in
            # the same way as the training dataset during the prediction
            self.x_min = x_min
            self.x_max = x_max

            # *************************************************************** #
            # Metrics initialization                                          #
            # Define a dictionary with the metrics as keys and empty lists as #
            # values. Used to store the metrics that will be plotted.         #
            # *************************************************************** #

            self.training_metrics = self.metrics_dictionary()
            self.validation_metrics = self.metrics_dictionary()

            # *************************************************************** #
            # Print the model architecture                                    #
            # *************************************************************** #

            layers_df = []
            for i in range(n_layers):
                layers_df.append(
                    {
                        "Layer name": "Input layer" if i == 0
                        else f"Hidden layer {i}" if i < n_layers - 1
                        else "Output layer",
                        "Number of neurons": n_neurons[i],
                        "Number of inputs": n_features if i == 0
                        else n_neurons[i - 1],
                        "Weights shape": self.layers[i].weights.shape,
                        "Bias shape": self.layers[i].biases.shape,
                        "Activation function": activations[i],
                    }
                )
            df = pandas.DataFrame(layers_df).T.to_string(header=False)
            print(f"\n{df}\n")

        except Exception as error:
            self.fatal_error("MultilayerPerceptron.__init__", error)

    # ********************************************************************** #
    #                                                                        #
    # ️ Multilayer Perceptron fit :                                           #
    #                                                                        #
    # ********************************************************************** #

    def fit(
        self,
        training_set,       # used in get_batch to avoid concatenation
        x_train_norm,       # X training set, training + metrics computation
        y_train,            # Y training set, training + metrics computation
        x_validation_norm,  # X validation set, metrics computation
        y_validation        # Y validation set, metrics computation
    ):

        # ******************************************************************* #
        # During the training phase, the parameters of the model (the weigths #
        # and the biases of each layer) will be updated.                      #
        # At each iteration, the model will :                                 #
        # - get a batch of the training set                                   #
        # - forward the batch through the network                             #
        # - compute the loss                                                  #
        # - backward the loss                                                 #
        # - update the parameters of the model                                #
        # ******************************************************************* #

        try:

            for epoch in ft_progress(range(self.epochs)):

                self.compute_metrics(
                        x_train_norm, y_train,
                        self.training_metrics
                )
                self.compute_metrics(
                        x_validation_norm, y_validation,
                        self.validation_metrics
                )

                for batch in range(self.n_batch):

                    x_batch, y_batch = self.get_batch(training_set)

                    output = self.forward(x_batch)
                    self.backward(output, y_batch)
                    self.optimize()

                self.optimizer.update_learning_rate()

        except Exception as error:
            self.fatal_error("MultilayerPerceptron.fit", error)

    # ********************************************************************** #
    #                                                                        #
    # ️ Multilayer Perceptron forward pass :                                  #
    #                                                                        #
    # ********************************************************************** #

    def forward(self, inputs):

        # ******************************************************************* #
        #  The forward pass is the process of computing the output of the     #
        #  network for a given input.                                         #
        #  In a multilayer perceptron, the forward pass is computed layer by  #
        #  layer.                                                             #
        #  For each layer :                                                   #
        #  - compute the weigthed sum by multiplying the input by the weights #
        #    of the layer and adding the bias.                                #
        #  - compute the output by applying the activation function to the    #
        #    weigthed sum.                                                    #
        #  - the output of a layer is the input of the next layer             #
        # ******************************************************************* #

        try:

            for layer in self.layers:
                weigthed_sum = layer.forward(inputs)
                output = layer.activation_function.forward(weigthed_sum)
                inputs = output

            return output

        except Exception as error:
            self.fatal_error("MultilayerPerceptron.forward", error)

    # ********************************************************************** #
    #                                                                        #
    # ️ Multilayer Perceptron backward pass :                                 #
    #                                                                        #
    # ********************************************************************** #

    def backward(self, output, y):

        # ******************************************************************* #
        # The backward pass is the process of computing the gradient of the   #
        # loss function with respect to the parameters of the model.          #
        # This gradient will be used to update the parameters of the model    #
        # during the optimisation phase.                                      #
        # The first step is to compute the gradient of the loss function with #
        # respect to the output of the network.                               #
        # Then, the gradient is propagated layer by layer in a reverse order. #
        # For each layer :                                                    #
        # - compute the gradient of the loss function with respect to the     #
        #   weigthed sum of the layer.                                        #
        # - compute the gradient of the loss function with respect to the     #
        #   parameters of the layer.                                          #
        # - compute the gradient of the loss function with respect to the     #
        #   input of the layer.                                               #
        # ******************************************************************* #

        try:

            # Compute the gradient of the loss function with respect to the
            # output of the network
            gradient = self.loss.backward(output, y)

            for layer in reversed(self.layers):

                # Compute the gradient of the loss function with respect to
                # the weigthed sum of the layer
                d_weigthed_sum = layer.activation_function.backward(gradient)

                # Compute the gradient of the loss function with respect to
                # the weigths, the biases and the input of the layer
                # Save them in the layer object to update the parameters later
                d_inputs = layer.backward(d_weigthed_sum)

                # The gradient of the loss function with respect to the input
                # of the layer is the gradient of the loss function with
                # respect to the output of the previous layer
                gradient = d_inputs

        except Exception as error:
            self.fatal_error("MultilayerPerceptron.backward", error)

    # ********************************************************************** #
    #                                                                        #
    # ️ Multilayer Perceptron parameters optimization :                       #
    #                                                                        #
    # ********************************************************************** #

    def optimize(self):

        # ******************************************************************* #
        #  The optimization phase is the process of updating the parameters   #
        #  of the model (the weigths and the biases of each layer)            #
        #  The parameters are updated using the gradient of the loss function #
        #  with respect to the parameters of the model.                       #
        #  The gradient of the loss function with respect to the parameters   #
        #  of the model is computed during the backward pass.                 #
        #  For each layer :                                                   #
        #  - update the weigths of the layer                                  #
        #  - update the biases of the layer                                   #
        # ******************************************************************* #

        try:

            for layer in self.layers:
                self.optimizer.update(layer)

        except Exception as error:
            self.fatal_error("MultilayerPerceptron.optimize", error)

    # ********************************************************************** #
    #                                                                        #
    # ️ Multilayer Perceptron prediction :                                    #
    #                                                                        #
    # ********************************************************************** #

    def predict(self, x):

        try:

            output = self.forward(x)
            y_hat = np.argmax(output, axis=0)
            return y_hat

        except Exception as error:
            self.fatal_error("MultilayerPerceptron.predict", error)

    # ********************************************************************** #
    #                                                                        #
    # ️ Multilayer Perceptron get-batch :                                     #
    #                                                                        #
    # ********************************************************************** #

    def get_batch(self, training_set):
        """
        Depending on the batch size and the number of samples in the training
        set, this function returns a batch of the training set.

        We can use a batch size of 1 to perform stochastic gradient descent.
        (Spped but unstable)
        If we use a batch size equal to the number of samples in the training
        set, we perform batch gradient descent. (Slow but stable)
        We can also use a batch size between 1 and the number of samples in the
        training set, default option = 32. (Speed and stability compromise)
        """
        try:

            n_samples = training_set.shape[0]

            # 0 is included, n_samples is excluded
            index_start = np.random.randint(0, n_samples)
            index_end = index_start + self.batch_size

            if index_end > n_samples:
                batch_begin = training_set[index_start:, :]
                batch_end = training_set[:index_end - n_samples, :]
                batch = np.concatenate((batch_begin, batch_end), axis=0)
            else:
                batch = training_set[index_start:index_end, :]

            x = batch[:, :-2]
            y = batch[:, -2:]
            return x.T, y.T

        except Exception as error:
            self.fatal_error("MultilayerPerceptron.get_batch", error)

    # ********************************************************************** #
    #                                                                        #
    # ️ Multilayer Perceptron utils functions :                               #
    #                                                                        #
    # ********************************************************************** #

    def metrics_dictionary(self):
        """
        Return a dictionary with the metrics as keys and empty lists as values.
        Used to store the metrics that will be plotted.
        """
        try:
            return {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
                "loss": []
            }
        except Exception as error:
            self.fatal_error("MultilayerPerceptron.metrics_dictionary", error)

    def compute_metrics(self, x, y, metrics):
        try:
            y_hat = self.predict(x)
            y = np.argmax(y, axis=0)
            metrics["loss"].append(self.loss.calculate(y_hat, y))
            metrics["accuracy"].append(accuracy_score_(y, y_hat))
            metrics["precision"].append(precision_score_(y, y_hat))
            metrics["recall"].append(recall_score_(y, y_hat))
            metrics["f1_score"].append(f1_score_(y, y_hat))
        except Exception as error:
            self.fatal_error(error)

    def save_model(self, path):
        try:
            with open(path, "wb") as file:
                pickle.dump(self, file)
        except Exception as error:
            self.fatal_error(error)

    def load_model(self, path):
        try:
            with open(path, "rb") as file:
                self = pickle.load(file)
            return self
        except Exception as error:
            self.fatal_error(error)

    def fatal_error(self, location, error_message):
        print(f"Error {location}: {error_message}")
        exit(1)
