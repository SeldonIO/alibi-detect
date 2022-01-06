from typing import Callable, Dict, Optional, Type, Union

import numpy as np
import tensorflow as tf
from alibi_detect.utils.tensorflow.prediction import (
    predict_batch, predict_batch_transformer)
from tensorflow.keras.layers import Dense, Flatten, Input, InputLayer
from tensorflow.keras.models import Model


class _Encoder(tf.keras.Model):
    def __init__(
            self,
            input_layer: Union[tf.keras.layers.Layer, tf.keras.Model],
            mlp: Optional[tf.keras.Model] = None,
            enc_dim: Optional[int] = None,
            step_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.input_layer = input_layer
        if isinstance(mlp, tf.keras.Model):
            self.mlp = mlp
        elif isinstance(enc_dim, int) and isinstance(step_dim, int):
            self.mlp = tf.keras.Sequential(
                [
                    Flatten(),
                    Dense(enc_dim + 2 * step_dim, activation=tf.nn.relu),
                    Dense(enc_dim + step_dim, activation=tf.nn.relu),
                    Dense(enc_dim, activation=None)
                ]
            )
        else:
            raise ValueError('Need to provide either `enc_dim` and `step_dim` or a '
                             'tf.keras.Sequential or tf.keras.Model `mlp`')

    def call(self, x: Union[np.ndarray, tf.Tensor, Dict[str, tf.Tensor]]) -> tf.Tensor:
        x = self.input_layer(x)
        return self.mlp(x)


class UAE(tf.keras.Model):
    def __init__(
            self,
            encoder_net: Optional[tf.keras.Model] = None,
            input_layer: Optional[Union[tf.keras.layers.Layer, tf.keras.Model]] = None,
            shape: Optional[tuple] = None,
            enc_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        is_enc = isinstance(encoder_net, tf.keras.Model)
        is_enc_dim = isinstance(enc_dim, int)
        if is_enc:
            self.encoder = encoder_net
        elif not is_enc and is_enc_dim:  # set default encoder
            input_layer = InputLayer(input_shape=shape) if input_layer is None else input_layer
            input_dim = np.prod(shape)
            step_dim = int((input_dim - enc_dim) / 3)
            self.encoder = _Encoder(input_layer, enc_dim=enc_dim, step_dim=step_dim)
        elif not is_enc and not is_enc_dim:
            raise ValueError('Need to provide either `enc_dim` or a tf.keras.Sequential'
                             ' or tf.keras.Model `encoder_net`.')

    def call(self, x: Union[np.ndarray, tf.Tensor, Dict[str, tf.Tensor]]) -> tf.Tensor:
        return self.encoder(x)


class HiddenOutput(tf.keras.Model):
    def __init__(
            self,
            model: tf.keras.Model,
            layer: int = -1,
            input_shape: tuple = None,
            flatten: bool = False
    ) -> None:
        super().__init__()
        if input_shape and not model.inputs:
            inputs = Input(shape=input_shape)
            model.call(inputs)
        else:
            inputs = model.inputs
        self.model = Model(inputs=inputs, outputs=model.layers[layer].output)
        self.flatten = Flatten() if flatten else tf.identity

    def call(self, x: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        return self.flatten(self.model(x))


def preprocess_drift(x: Union[np.ndarray, list], model: tf.keras.Model,
                     preprocess_batch_fn: Callable = None, tokenizer: Callable = None,
                     max_len: int = None, batch_size: int = int(1e10), dtype: Type[np.generic] = np.float32) \
        -> Union[np.ndarray, tf.Tensor]:
    """
    Prediction function used for preprocessing step of drift detector.

    Parameters
    ----------
    x
        Batch of instances.
    model
        Model used for preprocessing.
    preprocess_batch_fn
        Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
        processed by the TensorFlow model.
    tokenizer
        Optional tokenizer for text drift.
    max_len
        Optional max token length for text drift.
    batch_size
        Batch size.
    dtype
        Model output type, e.g. np.float32 or tf.float32.

    Returns
    -------
    Numpy array with predictions.
    """
    if tokenizer is None:
        return predict_batch(x, model, batch_size=batch_size, preprocess_fn=preprocess_batch_fn, dtype=dtype)
    else:
        return predict_batch_transformer(x, model, tokenizer, max_len, batch_size=batch_size, dtype=dtype)
