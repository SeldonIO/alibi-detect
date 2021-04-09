import numpy as np
import tensorflow as tf
from typing import Union


def predict_batch(x: Union[np.ndarray, tf.Tensor], model: tf.keras.Model, batch_size: int = int(1e10),
                  dtype: Union[np.float32, tf.DType] = np.float32) -> Union[np.ndarray, tf.Tensor]:
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
    dtype
        Model output type, e.g. np.float32 or tf.float32.

    Returns
    -------
    Numpy array or tensorflow tensor with model outputs.
    """
    n = x.shape[0]
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = isinstance(dtype, type)
    preds = []
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        preds_tmp = model(x[istart:istop])
        if return_np:
            preds_tmp = preds_tmp.numpy()
        preds.append(preds_tmp)
    if return_np:
        return np.concatenate(preds, axis=0)
    else:
        return tf.concat(preds, axis=0)


def predict_batch_transformer(x: np.ndarray, model: tf.keras.Model, tokenizer,
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

    Returns
    -------
    Numpy array or tensorflow tensor with model outputs.
    """
    n = x.shape[0]
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = isinstance(dtype, type)
    preds = []
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        tokens = tokenizer.batch_encode_plus(
            x[istart:istop], pad_to_max_length=True, max_length=max_len, return_tensors='tf'
        )
        preds_tmp = model(tokens)
        if return_np:
            preds_tmp = preds_tmp.numpy()
        preds.append(preds_tmp)
    if return_np:
        return np.concatenate(preds, axis=0)
    else:
        return tf.concat(preds, axis=0)
