import numpy as np


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Accuracy tells you the percentage of predictions that are accurate
    (i.e. the correct class was predicted).
    Accuracy doesn't give information about either error type.
    Args:
        y: a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """

    try:
        if not isinstance(y, np.ndarray) \
                or not isinstance(y_hat, np.ndarray):
            return None
        elif y.shape != y_hat.shape:
            return None
        elif y.size == 0:
            return None
        accuracy = np.mean(y == y_hat)
        return accuracy

    except Exception:
        return None
