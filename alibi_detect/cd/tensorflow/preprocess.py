import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, InputLayer
from tensorflow.keras.models import Model
from typing import Dict, Optional, Union
from alibi_detect.utils.tensorflow.prediction import predict_batch, predict_batch_transformer


class UAE(tf.keras.Model):
    def __init__(
            self,
            encoder_net: tf.keras.Sequential = None,
            input_layer: Union[tf.keras.layers.Layer, tf.keras.Model] = None,
            shape: Optional[tuple] = None,
            enc_dim: int = None
    ) -> None:
        super().__init__()
        is_tf_seq = isinstance(encoder_net, tf.keras.Sequential)
        is_enc_dim = isinstance(enc_dim, int)
        if is_tf_seq:
            self.encoder = encoder_net
        elif not is_tf_seq and is_enc_dim:  # set default encoder
            input_layer = InputLayer(input_shape=shape) if input_layer is None else input_layer
            input_dim = np.prod(shape)
            step_dim = int((input_dim - enc_dim) / 3)
            self.encoder = tf.keras.Sequential(
                [
                    input_layer,
                    Flatten(),
                    Dense(enc_dim + 2 * step_dim, activation=tf.nn.relu),
                    Dense(enc_dim + step_dim, activation=tf.nn.relu),
                    Dense(enc_dim, activation=None)
                ]
            )
        elif not is_tf_seq and not is_enc_dim:
            raise ValueError('Need to provide either `enc_dim` or a tf.keras.Sequential `encoder_net`.')

    def call(self, x: Union[np.ndarray, tf.Tensor, Dict[str, tf.Tensor]]) -> tf.Tensor:
        return self.encoder(x)


class HiddenOutput(tf.keras.Model):
    def __init__(
            self,
            model: tf.keras.Model,
            layer: int = -1,
            input_shape: tuple = None
    ) -> None:
        super().__init__()
        if input_shape and not model.inputs:
            inputs = Input(shape=input_shape)
            model.call(inputs)
        else:
            inputs = model.inputs
        self.model = Model(inputs=inputs, outputs=model.layers[layer].output)

    def call(self, x: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        return self.model(x)


def preprocess_drift(x: np.ndarray, model: tf.keras.Model, tokenizer=None,
                     max_len: int = None, batch_size: int = int(1e10),
                     dtype: type = np.float32) -> Union[np.ndarray, tf.Tensor]:
    """
    Prediction function used for preprocessing step of drift detector.

    Parameters
    ----------
    x
        Batch of instances.
    model
        Model used for preprocessing.
    tokenizer
        Optional tokenizer for text drift.
    max_len
        Optional max token length for text drift.
    batch_size
        Batch size.
    dtype
        Model output type, e.g. np.float32 or torch.float32.

    Returns
    -------
    Numpy array with predictions.
    """
    if tokenizer is None:
        return predict_batch(x, model, batch_size=batch_size, dtype=dtype)
    else:
        return predict_batch_transformer(x, model, tokenizer, max_len, batch_size=batch_size, dtype=dtype)
