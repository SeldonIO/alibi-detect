import logging
import numpy as np
import tensorflow as tf
from typing import Callable, Union
from alibi_detect.models.tensorflow.autoencoder import AE, AEGMM, Seq2Seq, VAE, VAEGMM

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
                  batch_size: int = int(1e10),
                  proba: bool = False,
                  return_class: bool = False,
                  n_categories: int = None,
                  shape: tuple = None,
                  dtype: type = np.float32
                  ) -> Union[np.ndarray, tuple]:
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
    return_class
        Whether to return model class predictions.
    n_categories
        Number of prediction categories. Can also be inferred from the model.
    shape
        Optional shape or tuple with shapes of the model predictions.
    dtype
        Output type.

    Returns
    -------
    Numpy array with predictions.
    """
    is_ae = isinstance(model, (AE, VAE))
    n = X.shape[0]
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
            shape = tuple([(n,) + p.shape[1:] if isinstance(p, np.ndarray) else
                           (n,) + p.numpy().shape[1:] for p in preds])
        else:
            shape = (n,) + preds.shape[1:] if isinstance(preds, np.ndarray) else (n,) + preds.numpy().shape[1:]

    if isinstance(shape[0], int):
        preds = np.zeros(shape, dtype=dtype)
    else:
        preds = tuple([np.zeros(s, dtype=dtype) for s in shape])

    n_minibatch = int(np.ceil(n / batch_size))
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        if isinstance(shape[0], tuple):
            preds_tmp = model(X[istart:istop])
            for j, p in enumerate(preds_tmp):
                preds[j][istart:istop] = p if isinstance(p, np.ndarray) else p.numpy()
        elif return_class:  # class predictions for classifier
            preds[istart:istop] = model(X[istart:istop]).numpy().argmax(axis=-1)
        else:
            preds[istart:istop] = model(X[istart:istop]).numpy()
    return preds


def predict_batch_transformer(model: tf.keras.Model,
                              tokenizer,
                              X: np.ndarray,
                              max_len: int,
                              batch_size: int = int(1e10)) -> np.ndarray:
    """

    Parameters
    ----------
    model
        HuggingFace transformer model.
    tokenizer
        Tokenizer for model.
    X
        Batch of instances.
    max_len
        Max token length.
    batch_size
        Batch size.

    Returns
    -------
    Numpy array with predictions.
    """
    n = X.shape[0]
    n_minibatch = int(np.ceil(n / batch_size))
    preds = []
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        x = tokenizer.batch_encode_plus(
            X[istart:istop],
            pad_to_max_length=True,
            max_length=max_len,
            return_tensors='tf'
        )
        preds_batch = model(x)
        preds.append(preds_batch.numpy())
    preds = np.concatenate(preds)
    return preds
