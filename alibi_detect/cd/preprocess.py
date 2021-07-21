import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from typing import Callable, Union


def classifier_uncertainty(
    x: Union[np.ndarray, list],
    model_fn: Callable,
    preds_type: str = 'probs',
    uncertainty_type: str = 'entropy',
    margin_width: float = 0.1,
) -> np.ndarray:
    """
    Evaluate model_fn on x and transform predictions to prediction uncertainties.

    Parameters
    ----------
    x
        Batch of instances.
    model_fn
        Function that evaluates a classification model on x in a single call (contains batching logic if necessary).
    preds_type
        Type of prediction output by the model. Options are 'probs' (in [0,1]) or 'logits' (in [-inf,inf]).
    uncertainty_type
        Method for determining the model's uncertainty for a given instance. Options are 'entropy' or 'margin'.
    margin_width
        Width of the margin if uncertainty_type = 'margin'. The model is considered uncertain on an instance
        if the highest two class probabilities it assigns to the instance differ by less than margin_width.

    Returns
    -------
    A scalar indication of uncertainty of the model on each instance in x.
    """

    preds = model_fn(x)

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
        diff = top_2_probs[:, 0] - top_2_probs[:, 1]
        uncertainties = (diff < margin_width).astype(int)
    else:
        raise NotImplementedError("Only uncertainty types 'entropy' or 'margin' supported")

    return uncertainties[:, None]  # Detectors expect N x d


def regressor_uncertainty(
    x: Union[np.ndarray, list],
    model_fn: Callable,
    uncertainty_type: str = 'mc_dropout',
    n_evals: int = 25,
) -> np.ndarray:
    """
    Evaluate model_fn on x and transform predictions to prediction uncertainties.

    Parameters
    ----------
    x
        Batch of instances.
    model_fn
        Function that evaluates a regression model on x in a single call (contains batching logic if necessary).
    uncertainty_type
        Method for determining the model's uncertainty for a given instance. Options are 'mc_dropout' or 'ensemble'.
        The former should output a scalar per instance. The latter should output a vector of predictions per instance.
    n_evals:
        The number of times to evaluate the model under different dropout configurations. Only relavent when using
        the 'mc_dropout' uncertainty type.

    Returns
    -------
    A scalar indication of uncertainty of the model on each instance in x.
    """

    if uncertainty_type == 'mc_dropout':
        preds = np.concatenate([model_fn(x) for _ in range(n_evals)], axis=-1)
    elif uncertainty_type == 'ensemble':
        preds = model_fn(x)
    else:
        raise NotImplementedError("Only 'mc_dropout' and 'ensemble' are supported uncertainty types for regressors.")

    uncertainties = np.std(preds, axis=-1)

    return uncertainties[:, None]  # Detectors expect N x d
