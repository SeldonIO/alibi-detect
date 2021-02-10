import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_class = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else np.round(y_true)
    y_pred_class = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else np.round(y_pred)
    return (y_true_class == y_pred_class).mean()
