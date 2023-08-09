import numpy as np


class Accuracy:

    def calculate(self, y_true, y_pred):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y_true)
        return accuracy
