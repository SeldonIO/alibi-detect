import logging
import numpy as np
import random
from typing import Dict, Callable, Optional, Tuple, Union
from alibi_detect.utils.sampling import reservoir_sampling

logger = logging.getLogger(__name__)


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


def encompass_batching(
    model: Callable,
    backend: str,
    batch_size: int,
    device: Optional[str] = None,
    preprocess_batch_fn: Optional[Callable] = None,
    tokenizer: Optional[Callable] = None,
    max_len: Optional[int] = None,
) -> Callable:
    """
    Takes a function that must be batch evaluated (on tokenized input) and returns a function
    that handles batching (and tokenization).
    """

    backend = backend.lower()
    kwargs = {'batch_size': batch_size, 'tokenizer': tokenizer, 'max_len': max_len,
              'preprocess_batch_fn': preprocess_batch_fn}
    if backend == 'tensorflow':
        from alibi_detect.cd.tensorflow.preprocess import preprocess_drift
    elif backend == 'pytorch':
        from alibi_detect.cd.pytorch.preprocess import preprocess_drift  # type: ignore
        kwargs['device'] = device
    else:
        raise NotImplementedError(f'{backend} not implemented. Use tensorflow or pytorch instead.')

    def model_fn(x: Union[np.ndarray, list]) -> np.ndarray:
        return preprocess_drift(x, model, **kwargs)  # type: ignore

    return model_fn


def encompass_shuffling_and_batch_filling(
    model_fn: Callable,
    batch_size: int
) -> Callable:
    """
    Takes a function that already handles batching but additionally performing shuffling
    and ensures instances are evaluated as part of full batches.
    """

    def new_model_fn(x: Union[np.ndarray, list]) -> np.ndarray:
        is_np = isinstance(x, np.ndarray)
        # shuffle
        n_x = len(x)
        perm = np.random.permutation(n_x)
        x = x[perm] if is_np else [x[i] for i in perm]
        # add extras if necessary
        final_batch_size = n_x % batch_size
        if final_batch_size != 0:
            doubles_inds = random.choices([i for i in range(n_x)], k=batch_size - final_batch_size)
            if is_np:
                x = np.concatenate([x, x[doubles_inds]], axis=0)  # type: ignore
            else:
                x += [x[i] for i in doubles_inds]
        # remove any extras and unshuffle
        preds = np.asarray(model_fn(x))[:n_x]
        preds = preds[np.argsort(perm)]
        return preds

    return new_model_fn


def get_input_shape(shape: Optional[Tuple], x_ref: Union[np.ndarray, list]) -> Optional[Tuple]:
    """ Optionally infer shape from reference data. """
    if isinstance(shape, tuple):
        return shape
    elif hasattr(x_ref, 'shape'):
        return x_ref.shape[1:]
    else:
        logger.warning('Input shape could not be inferred. '
                       'If alibi_detect.models.tensorflow.embedding.TransformerEmbedding '
                       'is used as preprocessing step, a saved detector cannot be reinitialized.')
        return None
