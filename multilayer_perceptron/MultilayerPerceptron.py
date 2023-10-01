# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    MultilayerPerceptron.py                           :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/26 14:41:05 by cmariot          #+#    #+#              #
#    Updated: 2023/10/01 11:44:56 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np
import pickle

from multilayer_perceptron.layer\
        import Layer
from multilayer_perceptron.Loss.binary_cross_entropy \
        import BinaryCrossEntropy_Loss
from multilayer_perceptron.Optimizers.sgd \
        import StandardGradientDescent
from multilayer_perceptron.ft_progress \
        import ft_progress
from multilayer_perceptron.Metrics.accuracy import accuracy_score_
from multilayer_perceptron.Metrics.precision import precision_score_
from multilayer_perceptron.Metrics.recall import recall_score_
from multilayer_perceptron.Metrics.f1_score import f1_score_


class MultilayerPerceptron:

    available_losses = {
        "binaryCrossentropy": BinaryCrossEntropy_Loss(),
    }

    available_optimizers = {
        "sgd": StandardGradientDescent,
    }

    def __init__(self,
                 n_neurons,
                 activations,
                 loss_name,
                 epochs,
                 batch_size,
                 x_min,
                 x_max,
                 n_train_samples,
                 learning_rate,
                 ):

        try:

            # Layers initialization
            self.layers = []
            for i in range(len(n_neurons)):
                self.layers.append(
                    Layer(
                        n_neurons[0] if i == 0 else n_neurons[i - 1],
                        n_neurons[i],
                        activations[i]
                    )
                )

            # Loss function initialization
            if loss_name not in self.available_losses:
                raise Exception("Loss function not available")
            self.loss = self.available_losses[loss_name]

            # Optimizer initialization
            self.optimizer = StandardGradientDescent(learning_rate)

            # Training parameters initialization
            self.epochs = epochs
            self.batch_size = batch_size
            self.n_batch = n_train_samples // batch_size
            if n_train_samples % batch_size != 0:
                self.n_batch += 1

            # Normalization parameters
            self.x_min = x_min
            self.x_max = x_max

            self.training_metrics = self.metrics_dictionary()
            self.validation_metrics = self.metrics_dictionary()

        except Exception as e:
            print(e)
            exit()

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
            print(error)
            exit()

    def forward(self, inputs):
        try:
            for layer in self.layers:
                weigthed_sum = layer.forward(inputs)
                output = layer.activation_function.forward(weigthed_sum)
                inputs = output
            return output
        except Exception as e:
            print(e)
            exit()

    def backward(self, y):
        try:
            dinputs = self.loss.backward(
                    self.layers[-1].activation_function.output, y
            )
            for layer in reversed(self.layers):
                layer.activation_function.backward(dinputs)
                layer.backward(layer.activation_function.dinputs)
                dinputs = layer.dinputs

        except Exception as e:
            print(e)
            exit()

    def optimize(self):
        self.learning_rates.append(self.optimizer.learning_rate)
        for layer in self.layers:
            self.optimizer.update(layer)

    def get_batch(self, training_set):
        """
        """
        try:
            n_samples = training_set.shape[0]
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
            return x, y
        except Exception as error:
            print(error)
            exit()

    def compute_metrics(self, x, y, metrics):
        y_pred = self.forward(x)
        metrics["loss"].append(self.loss.calculate(y_pred, y))
        y_hat = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        metrics["accuracy"].append(accuracy_score_(y, y_hat))
        metrics["precision"].append(precision_score_(y, y_hat))
        metrics["recall"].append(recall_score_(y, y_hat))
        metrics["f1_score"].append(f1_score_(y, y_hat))

    def fit(self,
            training_set,
            x_train_norm,
            y_train,
            x_validation_norm,
            y_validation
            ):
        for epoch in ft_progress(range(self.epochs)):
            for i in range(self.n_batch):
                x_batch, y_batch = self.get_batch(training_set)
                self.forward(x_batch)
                self.backward(y_batch)
                self.optimize()
            self.compute_metrics(
                    x_train_norm,
                    y_train,
                    self.training_metrics
            )
            self.compute_metrics(
                    x_validation_norm,
                    y_validation,
                    self.validation_metrics
            )

    def predict(self, x):
        y_pred = self.forward(x)
        y_hat = np.argmax(y_pred, axis=1)
        return y_hat

    def save_model(self, path):
        try:
            with open(path, "wb") as file:
                pickle.dump(self, file)
        except Exception as e:
            print(e)
            exit()

    def load_model(self, path):
        try:
            with open(path, "rb") as file:
                self = pickle.load(file)
            return self
        except Exception as e:
            print(e)
            exit()
