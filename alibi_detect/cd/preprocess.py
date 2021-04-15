import numpy as np
from sklearn.decomposition import PCA
from typing import Callable, Optional
from functools import partial


def pca(X: np.ndarray, n_components: int = 2, svd_solver: str = 'auto') -> np.ndarray:
    """
    Apply PCA dimensionality reduction and return the projection of X on
    the first `n_components` principal components.

    Parameters
    ----------
    X
        Batch of instances.
    n_components
        Number of principal component projections to return.
    svd_solver
        Solver used for SVD. Options are ‘auto’, ‘full’, ‘arpack’ or ‘randomized’.

    Returns
    -------
    Projection of X on first `n_components` principcal components.
    """
    X = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    pca.fit(X)
    X_pca = pca.transform(X)
    return X_pca


def classifier_uncertainty(
    X: np.ndarray,
    model: Callable,
    backend: Optional[str] = None,
    prediction_type: str = 'probs',
    uncertainty_type: str = 'entropy',
    margin_width: float = 0.1,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Evaluate model on X and transform predictions to prediction uncertainties.

    Parameters
    ----------
    X
        Batch of instances.
    model
        Classification model outputting class probabilities (or logits)
    backend
        Backend to use if model requires batch prediction. Options are 'tensorflow' or 'pytorch'.
    prediction_type
        Type of prediction output by the model. Options are 'probs' (in [0,1]) or 'logits' (in [-inf,inf]).
    uncertainty_type
        Method for determining the model's uncertainty for a given instance. Options are 'entropy' or 'margin'.
    margin_width
        Width of the margin if uncertainty_type = 'margin'. The model is considered uncertain on an instance
        if the highest two class probabilities it assigns to the instance differ by less than margin_width.
    batch_size
        Batch size used to evaluate model. Only relavent when backend has been specified for batch prediction.
    device
        Device type used. The default None tries to use the GPU and falls back on CPU if needed.
        Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.

    Returns
    -------
    Projection of X on first `n_components` principcal components.
    """

    if backend is not None:
        backend = backend.lower()
        if backend == 'tensorflow':
            from alibi_detect.cd.tensorflow.preprocess import preprocess_drift
            model_fn = partial(preprocess_drift, model=model, batch_size=batch_size)
        elif backend == 'pytorch':
            from alibi_detect.cd.pytorch.preprocess import preprocess_drift
            model_fn = partial(preprocess_drift, model=model, device=device, batch_size=batch_size)
        else:
            raise NotImplementedError(f'{backend} not implemented. Use tensorflow or pytorch instead.')
    else:
        model_fn = model
        if device not in [None, 'cpu']:
            raise NotImplementedError('Non-pytorch/tensorflow models must run on cpu')

    preds = np.asarray(model_fn(X))

    if prediction_type == 'probs':
        probs = preds
    elif prediction_type == 'logits':
        probs = preds.exp()/preds.exp().sum(-1)
    else:
        raise NotImplementedError("Only prediction types 'probs' and 'logits' supported.")

    if uncertainty_type == 'entropy':
        uncertainties = -(probs*np.log(probs)).sum(-1)[:, None]
    elif uncertainty_type == 'margin':
        top_2_probs = -np.partition(-probs, kth=1, axis=-1)[:, :2]
        diff = np.abs(top_2_probs[:, 0] - top_2_probs[:, 1])
        uncertainties = (diff < margin_width).astype(int)[:, None]
    else:
        raise NotImplementedError("Only uncertainty types 'entropy' or 'margin' supported")

    return uncertainties
