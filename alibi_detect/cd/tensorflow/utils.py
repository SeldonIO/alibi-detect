from typing import Callable
from functools import partial


def activate_train_mode_for_all_layers(model: Callable) -> Callable:
    model.trainable = False  # type: ignore
    model = partial(model, training=True)  # Note this affects batchnorm etc also
    return model
