import os
from pathlib import Path
import numpy as np
from typing import Union, List, Optional, Tuple, Callable, Dict, Any
from alibi_detect.utils._types import Literal
import matplotlib.pyplot as plt
from scipy.stats import uniform
import statsmodels.api as sm
import seaborn as sns
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def eval_detector(
        detector: type,  # TODO: type w/ BaseDriftDetector if introduced.
        X_ref: Union[np.ndarray, list],
        X_drift: Optional[Union[np.ndarray, list]] = None,
        C_ref: Optional[np.ndarray] = None,
        C_drift: Optional[np.ndarray] = None,
        roc_plot: bool = False,
        save_dir: Optional[os.PathLike] = None,
        verbose: bool = False,
        **kwargs: dict
) -> Dict[str, Any]:
    """

    Parameters
    ----------
    detector
    X_ref
    X_drift
    C_ref
    C_drift
    roc_plot
    save_dir
    verbose
    kwargs

    Returns
    -------

    """
    # Evaluate False Positive Rates
    if verbose:
        print('Computing False Positive Rates...')
    FPR, _ = eval_calibration(detector, X_ref, C=C_ref, verbose=verbose, save_dir=save_dir, **kwargs)
    results = {'FPR': FPR}

    # Evaluate True Positive Rates
    if verbose:
        print('Computing True Positive Rates...')
    TPR, _ = eval_test_power(detector, X_ref=X_ref, X_drift=X_drift, C_ref=C_ref, C_drift=C_drift,
                             verbose=verbose, save_dir=save_dir, **kwargs)
    results['TPR'] = TPR

    if len(FPR) > 1:
        AUC = compute_auc(FPR, TPR)
        results['AUC'] = AUC

    if roc_plot:
        save_file = Path(save_dir).joinpath('roc_plot.png') if save_dir is not None else None
        plot_roc(FPR, TPR, save_file=save_file)

    return results


def eval_calibration(
        detector: type,  # TODO: type w/ BaseDriftDetector if introduced.
        X: Union[np.ndarray, list],
        C: Optional[np.ndarray] = None,
        model: Optional[Callable] = None,
        detector_kwargs: Optional[dict] = None,
        n_runs: int = 100,
        n_samples: Tuple[int, int] = (500, 500),
        correction: Optional[Literal['bonferroni', 'fdr']] = None,
        sig_levels: Optional[Union[List[float], np.ndarray]] = None,
        qq_plot: bool = False,
        hist_plot: bool = False,
        save_dir: Optional[os.PathLike] = None,
        random_seed: Optional[Union[int, np.random.Generator]] = None,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO

    Parameters
    ----------
    detector
        The detector class to evaluate calibration for.
    X
        A reservoir of data. On each run instances are randomly subsampled to provide reference and test sets for
        the detector.
    C
        A reservoir of context data. On each run instances are randomly subsampled to provide reference and test sets
        for the detector. Should only be given for the :class:`~alibi_detect.cd.ContextMMDDrift` detector.
    model
        A PyTorch, TensorFlow or Sklearn model. Should only be given for detectors accepting a `model` arg, such
        as :class:`~alibi_detect.cd.ClassifierDrift`.
    detector_kwargs
        Dictionary of additional arguments and keyword arguments to pass to detector. If `detector` is a
        :class:`~alibi_detect.cd.ContextMMDDrift` type, a reservoir of context data should be passed via the `C` key.
        If `detector` is a type requiring a `model`, such as :class:`~alibi_detect.cd.ClassifierDrift`, this should be
        passed via the `model` field.
    n_runs
        The number of experiment runs. A larger number will give more stable p-value distributions, at the expense of
        a longer runtime.
    n_samples
        The number of samples to randomly sample from the data reservoir to create reference and test data for each run.
        The two data subsampled data sets will each contain `n_samples` instances, such that `2*n_samples` instances
        are sampled in total. TODO - update
    correction
        TODO
    sig_levels
        TODO
    qq_plot
        Whether to visualise the p-values on a Quantile-Quantile plot. This can also be done later with the
        :func:`~plot_qq` function.
    hist_plot
        Whether to visualise the p-values on a histogram. This can also be done later with the :func:`~plot_hist`
        function.
    random_seed
        Random seed (or :class:`~np.random.Generator`) to use for random sampling. If `None`, then fresh,
        unpredictable entropy will be pulled from the OS.
    verbose
        Whether to print a progress bar.

    Returns
    -------
    A NumPy ndarray containing the detector's p-value prediction for each of the `n_runs` number of runs.
    """
    # Check data sizes
    n_data = len(X)
    _check_sufficient_data_size(X, n_samples, n_runs)

    # Extract args and kwargs
    if detector_kwargs is None:
        detector_kwargs = {}
    detector_kwargs.pop('p_val', None)  # Set via sig_level

    # Perform preprocessing (if preprocess_fn exists) to save repeated compute in loop
    preprocess_fn = detector_kwargs.pop('preprocess_fn', None)
    if preprocess_fn is not None:
        X = preprocess_fn(X)

    # NumPy RNG
    rng = np.random.default_rng(random_seed)

    # Main experiment loop
    p_vals_list = []
    runs = tqdm(range(n_runs)) if verbose else range(n_runs)
    for _ in runs:
        # Subsample data
        idx = rng.choice(n_data, size=n_data, replace=False)
        idx_ref, idx_nodrift = idx[:n_samples[0]], idx[n_samples[0]:n_samples[0]+n_samples[1]]
        x_ref, x_nodrift = X[idx_ref], X[idx_nodrift]
        detector_args, predict_args = [x_ref], [x_nodrift]

        if C is not None:
            c_ref, c_nodrift = C[idx_ref], C[idx_nodrift]
            detector_args.append(c_ref)
            predict_args.append(c_nodrift)

        if model is not None:
            detector_args.append(model)

        # Init detector and predict
        dd = detector(*detector_args, **detector_kwargs)
        preds = dd.predict(*predict_args)
        p_vals_list.append(preds['data']['p_val'])
    p_vals = np.array(p_vals_list)

    # Apply univariate correction (if a univariate detector)
    if hasattr(dd, 'correction'):
        if correction is None:
            correction = getattr(dd, 'correction')
        p_vals = _multivariate_correction(p_vals, correction)

    # Apply significance tests
    if sig_levels is None:
        sig_levels = [getattr(dd, 'p_val')]
    FPR = np.array([np.sum(p_vals <= sig_levels[i]) / len(p_vals) for i in range(len(sig_levels))])

    # QQ-plot
    if qq_plot:
        save_file = Path(save_dir).joinpath('qq_plot.png') if save_dir is not None else None
        plot_qq(p_vals, save_file=save_file)
    # Histogram plot
    if hist_plot:
        save_file = Path(save_dir).joinpath('hist_plot.png') if save_dir is not None else None
        sig_level = sig_levels[0] if sig_levels is not None else None
        plot_hist(p_vals, save_file=save_file, sig_level=sig_level)

    return FPR, p_vals


def eval_test_power(
    detector: type,  # TODO: type w/ BaseDriftDetector if introduced.
    X_ref: Union[np.ndarray, list],
    X_drift: Union[np.ndarray, list],
    C_ref: Optional[np.ndarray] = None,
    C_drift: Optional[np.ndarray] = None,
    model: Optional[Callable] = None,
    detector_kwargs: Optional[dict] = None,
    correction: Optional[Literal['bonferroni', 'fdr']] = None,
    sig_levels: Union[List[float], np.ndarray] = np.linspace(0.01, 0.3, 10),
    n_runs: int = 100,
    n_samples: Tuple[int, int] = (500, 500),
    power_plot: bool = False,
    save_dir: Optional[os.PathLike] = None,
    random_seed: Optional[Union[int, np.random.Generator]] = None,
    verbose: bool = False,
) -> np.ndarray:
    """

    Parameters
    ----------

    Returns
    -------

    """
    if (C_ref is None) != (C_drift is None):
        raise ValueError("`C_ref` and `C_drift` must both be `None`, or both be given as np.ndarray's.")

    # Check data sizes
    n_ref, n_drift = len(X_ref), len(X_drift)
    _check_sufficient_data_size(X_ref, n_samples, n_runs, X_test=X_drift)

    # Setup detector_kwargs
    if detector_kwargs is None:
        detector_kwargs = {}

    # Perform preprocessing (if preprocess_fn exists) to save repeated compute in loop
    preprocess_fn = detector_kwargs.pop('preprocess_fn', None)
    if preprocess_fn is not None:
        X_ref = preprocess_fn(X_ref)
        X_drift = preprocess_fn(X_drift)

    # NumPy RNG
    rng = np.random.default_rng(random_seed)

    # Main experiment loop
    p_vals_list = []
    runs = tqdm(range(n_runs)) if verbose else range(n_runs)
    for _ in runs:
        # Subsample data
        idx_ref = rng.choice(n_ref, size=n_samples[0], replace=False)
        idx_drift = rng.choice(n_drift, size=n_samples[1], replace=False)
        x_ref, x_drift = X_ref[idx_ref], X_drift[idx_drift]
        detector_args, predict_args = [x_ref], [x_drift]

        if C_ref is not None:
            c_ref, c_drift = C_ref[idx_ref], C_drift[idx_drift]
            detector_args.append(c_ref)
            predict_args.append(c_drift)

        if model is not None:
            detector_args.append(model)

        # Init detector and predict
        dd = detector(*detector_args, **detector_kwargs)
        preds = dd.predict(*predict_args)
        p_vals_list.append(preds['data']['p_val'])
    p_vals = np.array(p_vals_list)

    # Apply univariate correction (if a univariate detector)
    if hasattr(dd, 'correction'):
        if correction is None:
            correction = getattr(dd, 'correction')
        p_vals = _multivariate_correction(p_vals, correction)

    # Apply significance tests
    if sig_levels is None:
        sig_levels = [getattr(dd, 'p_val')]
    power = np.array([np.sum(p_vals <= sig_levels[i]) / len(p_vals) for i in range(len(sig_levels))])

    if power_plot:
        save_file = Path(save_dir).joinpath('power_plot.png') if save_dir is not None else None
        plot_power(sig_levels, power, save_file=save_file)

    return power, p_vals


def plot_qq(p_vals: np.ndarray,
            title: Optional[str] = None,
            fig: Optional[plt.Axes] = None,
            save_file: Optional[os.PathLike] = None) -> plt.Figure:
    """
    Plot QQ-plots of p-value to evaluate drift detector calibration.

    Parameters
    ----------
    p_vals
    title
    fig

    Returns
    -------
    The matplotlib figure containing the 3x3 grid of axes.
    """
    if fig is None:
        fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 10))
    else:
        axs = fig.axes

    n = len(p_vals)
    for i in range(9):
        unifs = p_vals if i == 4 else np.random.rand(n)
        sm.qqplot(unifs, uniform(), line='45', ax=axs[i//3, i % 3])
        if i//3 < 2:
            axs[i//3, i % 3].set_xlabel('')
        if i % 3 != 0:
            axs[i//3, i % 3].set_ylabel('')

    if title is not None:
        fig.suptitle(title)

    if save_file is not None:
        _save_fig(save_file)

    return fig


def plot_hist(
    p_vals: Union[np.ndarray, List[np.ndarray]],
    title: Optional[str] = None,
    colors: Union[str, List[str]] = 'turquoise',
    labels: Optional[Union[str, List[str]]] = None,
    ylim: Optional[tuple] = None,
    binwidth: float = 0.05,
    sig_level: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    save_file: Optional[os.PathLike] = None
) -> plt.Axes:
    """
    Plot a histogram to evaluate drift detector calibration.

    Parameters
    ----------
    p_vals
    title
    colors
    labels
    ylim
    ax

    Returns
    -------
    The matplotlib axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(7, 5))

    p_vals = [p_vals] if not isinstance(p_vals, list) else p_vals
    colors = [colors] if not isinstance(colors, list) else colors
    labels = [labels] if not isinstance(labels, list) else labels
    for p_val, label, color in zip(p_vals, labels, colors):
        sns.histplot(p_val, color=color, label=label, binwidth=binwidth, stat='probability', ax=ax)
    if sig_level is not None:
        ax.axvline(sig_level, ls='--', color='firebrick', alpha=0.7)
    if label is not None:
        ax.legend(loc='upper right')
    ax.set_xlim(-0.02, 1.02)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel('Density')
    ax.set_xlabel('p-values')
    if title is not None:
        ax.set_title(title)

    if save_file is not None:
        _save_fig(save_file)

    return ax


def plot_power(
        sig_levels: np.ndarray,
        powers: Union[np.ndarray, List[np.ndarray]],
        title: Optional[str] = None,
        colors: Union[str, List[str]] = 'turquoise',
        labels: Optional[Union[str, List[str]]] = None,
        ax: Optional[plt.Axes] = None,
        save_file: Optional[os.PathLike] = None
) -> plt.Axes:
    """
    TODO
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(7, 5))

    powers = [powers] if not isinstance(powers, list) else powers
    colors = [colors] if not isinstance(colors, list) else colors
    labels = [labels] if not isinstance(labels, list) else labels
    for power, label, color in zip(powers, labels, colors):
        sns.lineplot(x=sig_levels, y=power, marker='o', mec='k', ms=8, color=color, label=label)
        ax.fill_between(sig_levels, power, alpha=0.2, color=color)
    if label is not None:
        ax.legend(loc='upper left')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(r'Significance level, $\alpha$')
    ax.set_ylabel('Test power')

    if save_file is not None:
        _save_fig(save_file)

    return ax


def plot_roc(
        FPR: Union[np.ndarray, List[np.ndarray]],
        TPR: Union[np.ndarray, List[np.ndarray]],
        title: Optional[str] = None,
        colors: Union[str, List[str]] = 'turquoise',
        labels: Optional[Union[str, List[str]]] = None,
        ax: Optional[plt.Axes] = None,
        save_file: Optional[os.PathLike] = None
) -> plt.Axes:
    """
    TODO
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(7, 5))

    FPRs = [FPR] if not isinstance(FPR, list) else FPR
    TPRs = [TPR] if not isinstance(TPR, list) else TPR
    colors = [colors] if not isinstance(colors, list) else colors
    labels = [labels] if not isinstance(labels, list) else labels
    max_val = 0.0
    for FPR, TPR, label, color in zip(FPRs, TPRs, labels, colors):
        sns.lineplot(x=FPR, y=TPR, marker='o', mec='k', ms=8, color=color, label=label, ax=ax)
        ax.fill_between(FPR, TPR, alpha=0.2, color=color)
        max_val = max(max_val, min(max(FPR), max(TPR)))
    sns.lineplot(x=[0, max_val], y=[0, max_val], linestyle='--', color='black', alpha=0.4, ax=ax)
    if label is not None:
        ax.legend(loc='upper left')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    if save_file is not None:
        _save_fig(save_file)

    return ax


def _multivariate_correction(p_vals: np.ndarray, correction: Literal['bonferroni', 'fdr']) -> np.ndarray:
    n_features = p_vals.shape[1]
    if correction == 'bonferroni':
        p_vals = np.min(p_vals, axis=1)*n_features
        p_vals = np.clip(p_vals, None, 1.0)  # Clipped as suggested by doi.org/10.1093/biomet/asaa027 (page 2)
    elif correction == 'fdr':
        p_vals_sorted = np.sort(p_vals, axis=1)
        p_vals = np.min(p_vals_sorted/(np.arange(n_features)+1), axis=1)*n_features
    return p_vals


def compute_auc(x: np.ndarray, y: np.ndarray):
    return np.trapz(y, x=x)  # could also use scipy.integrate.simpson or sklearn.metrics.auc


def _check_sufficient_data_size(X: Union[np.ndarray, list],
                                n_samples: Union[int, Tuple[int, int]],
                                n_runs: int,
                                X_test: Optional[Union[np.ndarray, list]] = None):
    pass
#    FACT = 0.5  # Factor to make less strict!
#    if X_test is None:
#        n_data = len(X)
#        if n_data/(2*n_samples) < FACT*n_runs:
#            warnings.warn('The size of the provided dataset might not be large enough to give reliable statistics.'
#                          'It is recommended that you provide more data. You can also try reducing `n_samples` or '
#                          '`n_runs`, but this could impact the statistics.')
#    else:
#        if len(X) / n_samples[0] < FACT*n_runs:
#            warnings.warn('The size of the provided reference set might not be large enough to give reliable '
#                          'statistics.')
#        if len(X_test) / n_samples[1] < FACT*n_runs:
#            warnings.warn('The size of the provided test set might not be large enough to give reliable '
#                          'statistics.')


def _save_fig(save_file: os.PathLike):
    # Create directory if needed
    save_dir = Path(save_file).parent
    if save_dir is not None:
        if not save_dir.is_dir():
            logger.warning('Directory {} does not exist and is now created.'.format(save_dir))
            save_dir.mkdir(parents=True, exist_ok=True)
    # Save figure
    plt.savefig(save_file)
