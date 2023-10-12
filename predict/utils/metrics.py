# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    metrics.py                                        :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/10/12 13:11:54 by cmariot          #+#    #+#              #
#    Updated: 2023/10/12 13:11:55 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import pandas
from multilayer_perceptron.Metrics.accuracy import accuracy_score_
from multilayer_perceptron.Metrics.precision import precision_score_
from multilayer_perceptron.Metrics.recall import recall_score_
from multilayer_perceptron.Metrics.f1_score import f1_score_


def print_metrics(y, y_hat, model):
    try:
        print(
            "\033[94m" +
            "\nMetrics computed on the test set :\n" +
            "\033[0m"
        )
        df = pandas.DataFrame(
            {
                "Binary cross entropy loss": model.loss.calculate(y_hat, y),
                "Accuracy": accuracy_score_(y, y_hat),
                "Recall": recall_score_(y, y_hat),
                "Precision": precision_score_(y, y_hat),
                "F1": f1_score_(y, y_hat)
            },
            index=["Test set"]
        )
        print(df.T, "\n")
    except Exception as error:
        print(error)
        exit()
