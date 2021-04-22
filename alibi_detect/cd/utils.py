import numpy as np
import random
from typing import Dict, Callable, Optional
from functools import partial
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


def get_preds(
    x: np.ndarray,
    model: Callable,
    backend: str,
    batch_size: int,
    shuffle: bool = False,
    force_full_batches: bool = False,
    device: Optional[str] = None,
    tokenizer: Optional[Callable] = None,
    max_len: Optional[int] = None,
) -> np.ndarray:

    backend = backend.lower()
    model_kwargs = {
        'model': model, 'batch_size': batch_size, 'tokenizer': tokenizer, 'max_len': max_len
    }
    if backend == 'tensorflow':
        from alibi_detect.cd.tensorflow.preprocess import preprocess_drift
    elif backend == 'pytorch':
        from alibi_detect.cd.pytorch.preprocess import preprocess_drift  # type: ignore
        model_kwargs['device'] = device
    else:
        raise NotImplementedError(f'{backend} not implemented. Use tensorflow or pytorch instead.')
    model_fn = partial(preprocess_drift, **model_kwargs)

    n_x = x.shape[0]

    if shuffle:
        perm = np.random.permutation(n_x)
        x = x[perm]

    final_batch_size = n_x % batch_size
    if force_full_batches and final_batch_size != 0:
        doubles_inds = random.choices([i for i in range(n_x)], k=batch_size - final_batch_size)
        x = np.concatenate([x, x[doubles_inds]], axis=0)

    preds = np.asarray(model_fn(x))[:n_x]

    if shuffle:
        preds = preds[np.argsort(perm)]

    return preds
