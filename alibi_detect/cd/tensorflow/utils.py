from typing import Callable
from functools import partial
import tensorflow as tf


def activate_train_mode_for_all_layers(model: tf.keras.Model) -> Callable:
    model.trainable = False
    model = partial(model, training=True)  # Note this affects batchnorm etc also
    return model
