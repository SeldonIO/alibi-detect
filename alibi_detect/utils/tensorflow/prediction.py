from functools import partial
import numpy as np
import tensorflow as tf
from typing import Callable, Union
from alibi_detect.utils.prediction import tokenize_transformer


def predict_batch(x: Union[list, np.ndarray, tf.Tensor], model: tf.keras.Model, batch_size: int = int(1e10),
                  preprocess_fn: Callable = None, dtype: Union[np.float32, tf.DType] = np.float32) \
        -> Union[np.ndarray, tf.Tensor]:
    """
    Make batch predictions on a model.

    Parameters
    ----------
    x
        Batch of instances.
    model
        tf.keras model or one of the other permitted types defined in Data.
    batch_size
        Batch size used during prediction.
    preprocess_fn
        Optional preprocessing function for each batch.
    dtype
        Model output type, e.g. np.float32 or tf.float32.

    Returns
    -------
    Numpy array or tensorflow tensor with model outputs.
    """
    n = len(x)
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = isinstance(dtype, type)
    preds = []
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        x_batch = x[istart:istop]
        if isinstance(preprocess_fn, Callable):  # type: ignore
            x_batch = preprocess_fn(x_batch)
        preds_tmp = model(x_batch)
        if return_np:
            preds_tmp = preds_tmp.numpy()
        preds.append(preds_tmp)
    if return_np:
        return np.concatenate(preds, axis=0)
    else:
        return tf.concat(preds, axis=0)


def predict_batch_transformer(x: Union[list, np.ndarray], model: tf.keras.Model, tokenizer: Callable,
                              max_len: int, batch_size: int = int(1e10),
                              dtype: Union[np.float32, tf.DType] = np.float32) \
        -> Union[np.ndarray, tf.Tensor]:
    """
    Make batch predictions using a transformers tokenizer and model.

    Parameters
    ----------
    x
        Batch of instances.
    model
        Transformer model.
    tokenizer
        Tokenizer for model.
    max_len
        Max token length.
    batch_size
        Batch size.
    dtype
        Model output type, e.g. np.float32 or tf.float32.

    Returns
    -------
    Numpy array or tensorflow tensor with model outputs.
    """
    preprocess_fn = partial(tokenize_transformer, tokenizer=tokenizer, max_len=max_len, backend='tf')
    return predict_batch(x, model, preprocess_fn=preprocess_fn, batch_size=batch_size, dtype=dtype)
