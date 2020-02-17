import logging
import numpy as np
import tensorflow as tf
from typing import Union
from alibi_detect.models.autoencoder import AE, VAE

logger = logging.getLogger(__name__)


def predict_batch(model: Union[AE, VAE, tf.keras.Model],
                  X: np.ndarray,
                  batch_size: int = 32,
                  proba: bool = False,
                  n_categories: int = None
                  ) -> np.ndarray:
    """
    Make batch predictions on a model.

    Parameters
    ----------
    model
        Autoencoder or tf.keras model.
    X
        Batch of instances.
    batch_size
        Batch size used during prediction.
    proba
        Whether to return model prediction probabilities.
    n_categories
        Number of prediction categories. Can also be inferred from the model.

    Returns
    -------
    Numpy array with predictions.
    """
    n = X.shape[0]
    is_ae = isinstance(model, (AE, VAE))
    if is_ae:
        shape = X.shape
        dtype = np.float32
    elif proba:
        n_categories = n_categories if n_categories else model(X[0:1]).shape[-1]
        shape = (n, n_categories)
        dtype = np.float32
    else:
        shape = (n,)
        dtype = np.int64
    preds = np.zeros(shape, dtype=dtype)
    n_minibatch = int(np.ceil(n / batch_size))
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        if is_ae or proba:
            preds[istart:istop] = model(X[istart:istop]).numpy()
        else:
            preds[istart:istop] = model(X[istart:istop]).numpy().argmax(axis=-1)
    return preds
