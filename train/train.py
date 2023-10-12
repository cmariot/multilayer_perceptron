# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    train.py                                          :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/26 13:59:17 by cmariot          #+#    #+#              #
#    Updated: 2023/10/03 09:44:49 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #


from multilayer_perceptron.header import header
from utils.parse_arguments import parse_arguments
from utils.get_datasets import (get_training_data, get_validation_data)
from multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron
from utils.plots import (
    print_metrics, plot_loss, plot_metrics, plot_loss_and_metrics
)



if __name__ == "__main__":

    header()

    (
        train_path,       # Path to the training dataset
        validation_path,  # Path to the validation dataset
        n_neurons,        # Number of neurons in each layer
        activations,      # Activation function in each layer
        loss_name,        # Loss function,
        optimizer_name,   # Optimizer function
        epochs,           # Number of epochs
        batch_size,       # Batch size
        learning_rate,    # Initial learning rate
        decay,            # Decay
        momentum,         # Momentum
        early_stopping    # Number of epochs without improvement before stopping
    ) = parse_arguments()

    # ########################################################### #
    # Load the datasets :                                         #
    # - Training dataset is used to train the model               #
    # - Validation dataset is used to check the model performance #
    #                                                             #
    # The dataset features are normalized (between 0 and 1)       #
    #                                                             #
    # The dataset targets are replaced by 0 for malignant and     #
    # 1 for benign.                                               #
    # ########################################################### #

    (
        x_train_norm,
        y_train,
        training_set,
        x_min,
        x_max
    ) = get_training_data(train_path)

    (
        x_validation_norm,
        y_validation
    ) = get_validation_data(validation_path, x_min, x_max)

    # ########################################################### #
    # Create the neural network model :                           #
    # - Input layer                                               #
    # - Hidden layer 1                                            #
    # - Hidden layer 2                                            #
    # - Output layer                                              #
    # ########################################################### #

    # Create the multilayer perceptron object
    model = MultilayerPerceptron(
        n_neurons=n_neurons,
        activations=activations,
        loss_name=loss_name,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        decay=decay,
        momentum=momentum,
        epochs=epochs,
        batch_size=batch_size,
        train_set_shape=x_train_norm.shape,
        early_stopping=early_stopping,
        x_min=x_min,
        x_max=x_max,
    )

    # ############### #
    # Train the model #
    # ############### #

    input("Press Enter to start training the model ...\n")

    model.fit(
        training_set,
        x_train_norm,
        y_train,
        x_validation_norm,
        y_validation
    )

    model.save_model("../model.pkl")
    model.save_metrics("../metrics.csv")

    # ############### #
    # Test the model  #
    # ############### #

    # Print the metrics of the training and validation sets
    print_metrics(model.training_metrics, model.validation_metrics)

    # Plot the loss of the training and validation sets on the same graph
    plot_loss(model.training_metrics, model.validation_metrics)

    # Plot the accuracy, the precision, the recall and the f1-score of
    # the training and validation sets
    plot_metrics(model.training_metrics, model.validation_metrics)

    # Plot the loss and the metrics of the training and validation sets on the
    # same graph
    plot_loss_and_metrics(model.training_metrics, model.validation_metrics)

    # 3d plot of the validation set, two first features and the target
    import matplotlib.pyplot as plt
    import numpy as np

    y_hat = model.predict(x_validation_norm)
    y_validation = np.argmax(y_validation, axis=0).reshape(-1, 1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    colors = {
        "true positive": "green",
        "true negative": "red",
        "false positive": "blue",
        "false negative": "yellow"
    }
    validation_colors = []
    print(y_validation.shape)
    for i in range(len(y_validation)):
        if y_validation[i] == 0 and y_hat[i] == 0:
            validation_colors.append(colors["true positive"])
        elif y_validation[i] == 1 and y_hat[i] == 1:
            validation_colors.append(colors["true negative"])
        elif y_validation[i] == 0 and y_hat[i] == 1:
            validation_colors.append(colors["false negative"])
        elif y_validation[i] == 1 and y_hat[i] == 0:
            validation_colors.append(colors["false positive"])

    ax.scatter(
        x_validation_norm[2],
        x_validation_norm[4],
        x_validation_norm[5],
        c=validation_colors
    )

    plt.show()