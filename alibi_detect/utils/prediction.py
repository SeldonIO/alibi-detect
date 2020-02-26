import logging
import numpy as np
import tensorflow as tf
from typing import Callable, Union
from alibi_detect.models.autoencoder import AE, AEGMM, Seq2Seq, VAE, VAEGMM

logger = logging.getLogger(__name__)

Data = Union[
    tf.keras.Model,
    Callable,
    AE,
    AEGMM,
    Seq2Seq,
    VAE,
    VAEGMM
]


def predict_batch(model: Data,
                  X: np.ndarray,
                  batch_size: int = 32,
                  proba: bool = False,
                  return_class: bool = False,
                  n_categories: int = None,
                  shape: tuple = None
                  ) -> np.ndarray:
    """
    Make batch predictions on a model.

    Parameters
    ----------
    model
        tf.keras model or one of the other permitted types defined in Data.
    X
        Batch of instances.
    batch_size
        Batch size used during prediction.
    proba
        Whether to return model prediction probabilities.
    n_categories
        Number of prediction categories. Can also be inferred from the model.
    shape
        Optional shape or tuple with shapes of the model predictions.

    Returns
    -------
    Numpy array with predictions.
    """
    is_ae = isinstance(model, (AE, VAE))
    n = X.shape[0]
    dtype = np.float32
    if isinstance(shape, tuple):
        pass  # already defined shape
    elif is_ae:
        shape = X.shape
    elif proba:
        n_categories = n_categories if n_categories else model(X[0:1]).shape[-1]
        shape = (n, n_categories)
    elif return_class:
        shape = (n,)
        dtype = np.int64
    else:
        preds = model(X[0:1])
        if isinstance(preds, tuple):
            shape = tuple([(n, p.shape[1:]) for p in preds])
        else:
            shape = (n, preds.shape[1:])

    if isinstance(shape[0], tuple):
        preds = tuple([np.zeros(s, dtype=dtype) for s in shape])
    else:
        preds = np.zeros(shape, dtype=dtype)

    n_minibatch = int(np.ceil(n / batch_size))
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        if is_ae or proba:
            preds[istart:istop] = model(X[istart:istop]).numpy()
        elif isinstance(shape[0], tuple):
            preds_tmp = model(X[istart:istop])
            for j, p in enumerate(preds_tmp):
                preds[j][istart:istop] = p.numpy()
        else:  # class predictions for classifier
            preds[istart:istop] = model(X[istart:istop]).numpy().argmax(axis=-1)
    return preds
