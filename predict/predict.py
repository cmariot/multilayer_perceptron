# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    predict.py                                        :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:21:00 by cmariot          #+#    #+#              #
#    Updated: 2023/10/05 21:14:50 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

from multilayer_perceptron.header import header
from utils.parse_args import parse_arguments
from utils.load_files import load_model, load_dataset, save_predictions
from utils.metrics import print_metrics
from utils.plot import plot_confusion_matrix

from multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron


if __name__ == "__main__":

    header()

    # Parse command line arguments
    (
        predict_path,  # Path to the prediction dataset
        model_path     # Path to the model
    ) = parse_arguments()

    # Load the model
    model: MultilayerPerceptron = load_model(model_path)
    x_min = model.x_min
    x_max = model.x_max

    # Load the dataset and normalize the features
    x, x_norm, y = load_dataset(predict_path, x_min, x_max)

    # Predict the classes on the test set
    y_hat = model.predict(x_norm)

    if y is not None:

        # Print the metrics computed on the test set
        print_metrics(y, y_hat, model)

        # Plot the confusion matrix on the test set
        plot_confusion_matrix(y, y_hat)

    # Save the predictions in a csv file
    save_predictions(y_hat, "../datasets/predictions.csv")
