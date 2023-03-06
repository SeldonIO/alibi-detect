from functools import partial
from typing import Callable


def activate_train_mode_for_all_layers(model: Callable) -> Callable:
    model.trainable = False  # type: ignore
    model = partial(model, training=True)  # Note this affects batchnorm etc also
    return model
