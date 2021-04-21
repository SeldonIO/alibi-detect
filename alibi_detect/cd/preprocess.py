import numpy as np
from sklearn.decomposition import PCA
from scipy.special import softmax
from scipy.stats import entropy
from typing import Callable, Optional
from alibi_detect.cd.utils import activate_train_mode_for_dropout_layers, get_preds


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
    x: np.ndarray,
    model: Callable,
    backend: Optional[str] = None,
    preds_type: str = 'probs',
    uncertainty_type: str = 'entropy',
    margin_width: float = 0.1,
    batch_size: int = 32,
    device: Optional[str] = None,
    tokenizer: Optional[Callable] = None,
    max_len: Optional[int] = None,
) -> np.ndarray:
    """
    Evaluate model on x and transform predictions to prediction uncertainties.

    Parameters
    ----------
    x
        Batch of instances.
    model
        Classification model outputting class probabilities (or logits)
    backend
        Backend to use if model requires batch prediction. Options are 'tensorflow' or 'pytorch'.
    preds_type
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
    tokenizer
        Optional tokenizer for NLP models.
    max_len
        Optional max token length for NLP models.

    Returns
    -------
    A scalar indication of uncertainty of the model on each instance in x.
    """

    if backend is not None:
        preds = get_preds(
            x, model, backend, batch_size, device=device, tokenizer=tokenizer, max_len=max_len
        )
    else:
        if device not in [None, 'cpu']:
            raise NotImplementedError('Non-pytorch/tensorflow models must run on cpu')
        preds = np.asarray(model(x))

    if preds_type == 'probs':
        if np.abs(1 - np.sum(preds, axis=-1)).mean() > 1e-6:
            raise ValueError("Probabilities across labels should sum to 1")
        probs = preds
    elif preds_type == 'logits':
        probs = softmax(preds, axis=-1)
    else:
        raise NotImplementedError("Only prediction types 'probs' and 'logits' supported.")

    if uncertainty_type == 'entropy':
        uncertainties = entropy(probs, axis=-1)
    elif uncertainty_type == 'margin':
        top_2_probs = -np.partition(-probs, kth=1, axis=-1)[:, :2]
        diff = np.abs(top_2_probs[:, 0] - top_2_probs[:, 1])
        uncertainties = (diff < margin_width).astype(int)
    else:
        raise NotImplementedError("Only uncertainty types 'entropy' or 'margin' supported")
    
    return uncertainties[:, None]  # Detectors expect N x d


def regressor_uncertainty(
    x: np.ndarray,
    model: Callable,
    backend: str,
    uncertainty_type: str = 'mc_dropout',
    n_evals: int = 25,
    batch_size: int = 32,
    device: Optional[str] = None,
    tokenizer: Optional[Callable] = None,
    max_len: Optional[int] = None,
) -> np.ndarray:
    """
    Evaluate model on x and transform predictions to prediction uncertainties.

    Parameters
    ----------
    x
        Batch of instances.
    model
        Classification model outputting class probabilities (or logits)
    backend
        Backend to use if model requires batch prediction. Options are 'tensorflow' or 'pytorch'.
    uncertainty_type
        Method for determining the model's uncertainty for a given instance. Options are 'mc_dropout' or 'ensemble'.
        The former should output a scalar per instance. The latter should output a vector of predictions per instance.
    n_evals:
        The number of times to evaluate the model under different dropout configurations. Only relavent when using
        the 'mc_dropout' uncertainty type.
    batch_size
        Batch size used to evaluate model. Only relavent when backend has been specified for batch prediction.
    device
        Device type used. The default None tries to use the GPU and falls back on CPU if needed.
        Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
    tokenizer
        Optional tokenizer for NLP models.
    max_len
        Optional max token length for NLP models.

    Returns
    -------
    A scalar indication of uncertainty of the model on each instance in xs.
    """

    if uncertainty_type == 'mc_dropout':
        model = activate_train_mode_for_dropout_layers(model, backend)
        preds = np.concatenate([
            get_preds(
                x, model, backend, batch_size, shuffle=True, force_full_batches=True,
                device=device, tokenizer=tokenizer, max_len=max_len
            ) for i in range(n_evals)
        ], axis=-1)
    elif uncertainty_type == 'ensemble':
        if backend is not None:
            preds = get_preds(
                x, model, backend, batch_size, device=device, tokenizer=tokenizer, max_len=max_len
            )
        else:
            if device not in [None, 'cpu']:
                raise NotImplementedError('Non-pytorch/tensorflow models must run on cpu')
            preds = np.asarray(model(x))
    else:
        raise NotImplementedError("Only 'mc_dropout' and 'ensemble' are supported uncertainty types for regressors.")

    uncertainties = np.std(preds, axis=-1)

    return uncertainties[:, None]  # Detectors expect N x d