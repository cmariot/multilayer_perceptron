import pandas
from Metrics.accuracy import accuracy_score_
from Metrics.f1_score import f1_score_
from Metrics.precision import precision_score_
from Metrics.recall import recall_score_


def metrics_dictionary():
    """
    Return a dictionary with the metrics as keys and empty lists as values.
    Used to store the metrics that will be plotted.
    """
    return {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }


def compute_metrics(metrics_dictionary, y, y_hat):
    """
    Use the metrics functions to compute the metrics and append them to the
    metrics_dictionary.
    """
    metrics_functions = [
        accuracy_score_, precision_score_, recall_score_, f1_score_
    ]
    for i, list_ in enumerate(metrics_dictionary.values()):
        list_.append(metrics_functions[i](y, y_hat))


def get_batch(x, y, i, batch_size):
    start = i * batch_size
    end = (i + 1) * batch_size
    if end > x.shape[0]:
        end = x.shape[0]
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def print_final_metrics(metrics_dictionary):
    final_validation_metrics = {
        "accuracy": metrics_dictionary["accuracy"][-1],
        "precision": metrics_dictionary["precision"][-1],
        "recall": metrics_dictionary["recall"][-1],
        "f1_score": metrics_dictionary["f1_score"][-1],
    }

    df_metrics = pandas.DataFrame(
        final_validation_metrics,
        index=[None],
    )

    # Simple description of the metrics
    description = {
        "accuracy": "(TP + TN) / total",
        "precision": "(TP) / (TP + FP)",
        "recall": "(TP) / (TP + FN)",
        "f1_score": "(2 * precision * recall) / (precision + recall)",
    }
    # Add a description row to the dataframe
    df_metrics = df_metrics.append(description, ignore_index=True)

    df_metrics = df_metrics.transpose()
    print("Metrics on validation set:\n")
    print(df_metrics.to_string(header=False))
