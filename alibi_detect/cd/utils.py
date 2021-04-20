import numpy as np
from typing import Dict, Callable
from functools import partial
from torch import nn
from alibi_detect.utils.sampling import reservoir_sampling


def update_reference(X_ref: np.ndarray,
                     X: np.ndarray,
                     n: int,
                     update_method: Dict[str, int] = None,
                     ) -> np.ndarray:
    """
    Update reference dataset for drift detectors.

    Parameters
    ----------
    X_ref
        Current reference dataset.
    X
        New data.
    n
        Count of the total number of instances that have been used so far.
    update_method
        Dict with as key `reservoir_sampling` or `last` and as value n. `reservoir_sampling` will apply
        reservoir sampling with reservoir of size n while `last` will return (at most) the last n instances.

    Returns
    -------
    Updated reference dataset.
    """
    if isinstance(update_method, dict):
        update_type = list(update_method.keys())[0]
        size = update_method[update_type]
        if update_type == 'reservoir_sampling':
            return reservoir_sampling(X_ref, X, size, n)
        elif update_type == 'last':
            X_update = np.concatenate([X_ref, X], axis=0)
            return X_update[-size:]
        else:
            raise KeyError('Only `reservoir_sampling` and `last` are valid update options for X_ref.')
    else:
        return X_ref


def activate_train_mode_for_dropout_layers(model: Callable, backend: str) -> Callable:
    # TODO: Figure a way to do this properly for tensorflow models.
    if backend == 'pytorch':
        model.eval()
        n_dropout_layers = 0
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                n_dropout_layers += 1
        if n_dropout_layers == 0:
            raise ValueError("No dropout layers identified.")
        else:
            print(f'{n_dropout_layers} identified')
    elif backend == 'tensorflow':
        model.trainable = False
        model = partial(model, training=True)  # Note this affects batchnorm etc also
    else:
        raise NotImplementedError("Only 'pytorch' or 'tensorflow' backends supported")

    return model
