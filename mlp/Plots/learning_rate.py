# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    learning_rate.py                                  :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:48 by cmariot          #+#    #+#              #
#    Updated: 2023/08/24 14:39:49 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import matplotlib.pyplot as plt


def plot_learning_rate(learning_rates):
    """
    Plot the learning rate evolution.
    """

    try:

        plt.plot(learning_rates)
        plt.title("Learning rate evolution")
        plt.ylabel("Learning rate")
        plt.xlabel("Epochs")
        plt.grid()
        plt.show()

    except Exception as e:

        print(f"Error while plotting the learning rate: {e}")
        exit()
