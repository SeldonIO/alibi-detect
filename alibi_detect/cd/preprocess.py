import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras.models import Model
from alibi_detect.models.autoencoder import EncoderAE
from alibi_detect.utils.prediction import predict_batch

logger = logging.getLogger(__name__)


def uae(X: np.ndarray,
        encoder_net: tf.keras.Sequential = None,
        enc_dim: int = None,
        batch_size: int = int(1e10)) -> np.ndarray:
    """
    Dimensionality reduction with an untrained autoencoder.

    Parameters
    ----------
    X
        Batch of instances.
    encoder_net
        Encoder network as a tf.keras.Sequential model.
    enc_dim
        Alternatively, only the dimension of the encoding can be provided and
        a default network with 2 hidden layers is constructed.
    batch_size
        Batch size used when making predictions with the autoencoder.

    Returns
    -------
    Encoded batch of instances.
    """
    is_tf_seq = isinstance(encoder_net, tf.keras.Sequential)
    is_enc_dim = isinstance(enc_dim, int)
    if not is_tf_seq and is_enc_dim:  # set default encoder
        input_dim = np.prod(X.shape[1:])
        step_dim = int((input_dim - enc_dim) / 3)
        encoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=X.shape[1:]),
                Flatten(),
                Dense(enc_dim + 2 * step_dim, activation=tf.nn.relu),
                Dense(enc_dim + step_dim, activation=tf.nn.relu),
                Dense(enc_dim, activation=None)
            ]
        )
    elif not is_tf_seq and not is_enc_dim:
        raise ValueError('Need to provide either `enc_dim` or a tf.keras.Sequential `encoder_net`.')
    enc = EncoderAE(encoder_net)
    X_enc = predict_batch(enc, X, batch_size=batch_size)
    return X_enc


def hidden_output(X: np.ndarray,
                  model: tf.keras.Model = None,
                  layer: int = -2,
                  batch_size: int = int(1e10)) -> np.ndarray:
    """
    Return hidden layer output from a model on a batch of instances.

    Parameters
    ----------
    X
        Batch of instances.
    model
        tf.keras.Model.
    layer
        Hidden layer of model to use as output. The default of -2 would refer to the logits layer.
    batch_size
        Batch size used for the model predictions.

    Returns
    -------
    Model predictions using the specified hidden layer as output layer.
    """
    hidden_model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
    X_hidden = predict_batch(hidden_model, X, batch_size=batch_size)
    return X_hidden
