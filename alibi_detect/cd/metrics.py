import numpy as np
from typing import Union, List, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from scipy.stats import uniform
import statsmodels.api as sm
import seaborn as sns
#  import warnings
from tqdm import tqdm


def eval_calibration(
        detector: type,  # TODO: type w/ BaseDriftDetector if introduced.
        X: Union[np.ndarray, list],
        C: Optional[np.ndarray] = None,
        model: Optional[Callable] = None,
        detector_kwargs: Optional[dict] = None,
        n_runs: int = 100,
        n_samples: int = 500,
        qq_plot: bool = False,
        hist_plot: bool = False,
        random_seed: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
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
        Keyword arguments to pass to detector.
    n_runs
        The number of experiment runs. A larger number will give more stable p-value distributions, at the expense of
        a longer runtime.
    n_samples
        The number of samples to randomly sample from the data reservoir to create reference and test data for each run.
        The two data subsampled data sets will each contain `n_samples` instances, such that `2*n_samples` instances
        are sampled in total.
    qq_plot
        Whether to visualise the p-values on a Quantile-Quantile plot. This can also be done later with the
        :func:`~plot_qq` function.
    hist_plot
        Whether to visualise the p-values on a histogram. This can also be done later with the :func:`~plot_hist`
        function.
    random_seed
        Random seed (or :class:`~np.random.Generator`) to use for random sampling. If `None`, then fresh,
        unpredictable entropy will be pulled from the OS.

    Returns
    -------
    A numpy array containing the detector's p-value prediction for each of the `n_runs` number of runs.
    """
    # Check data sizes
    n_data = len(X)
    _check_sufficient_data_size(X, n_samples, n_runs)

    # Setup detector_kwargs
    if detector_kwargs is None:
        detector_kwargs = {}

    # Perform preprocessing (if preprocess_fn exists) to save repeated compute in loop
    preprocess_fn = detector_kwargs.pop('preprocess_fn', None)
    if preprocess_fn is not None:
        X = preprocess_fn(X)

    # NumPy RNG
    rng = np.random.default_rng(random_seed)

    # Main experiment loop
    p_vals_list = []
    for _ in tqdm(range(n_runs)):
        # Subsample data
        idx = rng.choice(n_data, size=n_data, replace=False)
        idx_ref, idx_nodrift = idx[:n_samples], idx[n_samples:2*n_samples]
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
    p_vals = np.array(p_vals_list).flatten()
    # TODO - is above OK? (to flatten over features for univariate detectors)

    # QQ-plot
    if qq_plot:
        plot_qq(p_vals)
    # Histogram plot
    if hist_plot:
        plot_hist(p_vals)
    # Show plots
    if qq_plot or hist_plot:
        plt.show()

    return p_vals


def eval_test_power(
    detector: type,  # TODO: type w/ BaseDriftDetector if introduced.
    X_ref: Union[np.ndarray, list],
    X_drift: Union[np.ndarray, list],
    C_ref: Optional[np.ndarray] = None,
    C_drift: Optional[np.ndarray] = None,
    model: Optional[Callable] = None,
    detector_kwargs: Optional[dict] = None,
    sig_levels: np.ndarray = np.linspace(0.05, 0.5, 10),
    n_runs: int = 100,
    n_samples: Tuple[int, int] = (500, 500),
    return_auc: bool = True,
    power_plot: bool = False,
    random_seed: Optional[Union[int, np.random.Generator]] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
    detector_kwargs.pop('p_val', None)  # Need to set this ourselves

    # Perform preprocessing (if preprocess_fn exists) to save repeated compute in loop
    preprocess_fn = detector_kwargs.pop('preprocess_fn', None)
    if preprocess_fn is not None:
        X_ref = preprocess_fn(X_ref)
        X_drift = preprocess_fn(X_drift)

    # NumPy RNG
    rng = np.random.default_rng(random_seed)

    # Main experiment loop
    power = np.zeros(len(sig_levels), dtype=float)
    for _ in tqdm(range(n_runs)):
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
        for i, sig in enumerate(sig_levels):
            dd = detector(*detector_args, p_val=sig, **detector_kwargs)
            preds = dd.predict(*predict_args)
            power[i] += preds['data']['is_drift']
    power /= n_runs

    if power_plot:
        plot_power(sig_levels, power)
        plt.show()

    if return_auc:
        auc = compute_auc(sig_levels, power)
        return power, auc

    return power


def plot_qq(p_vals: np.ndarray,
            title: Optional[str] = None,
            fig: Optional[plt.Axes] = None) -> plt.Figure:
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

    return fig


def plot_hist(
    p_vals: Union[np.ndarray, List[np.ndarray]],
    title: Optional[str] = None,
    colors: Union[str, List[str]] = 'turquoise',
    labels: Optional[Union[str, List[str]]] = None,
    ylim: Optional[tuple] = None,
    binwidth: float = 0.05,
    ax: Optional[plt.Axes] = None
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
        sns.histplot(p_val, color=color, kde=True, label=label, binwidth=binwidth, stat='probability', ax=ax)
    if label is not None:
        ax.legend(loc='upper right')
    ax.set_xlim(-0.02, 1.02)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel('Density')
    ax.set_xlabel('p-values')
    if title is not None:
        ax.set_title(title)

    return ax


def plot_power(
        sig_levels: np.ndarray,
        powers: Union[np.ndarray, List[np.ndarray]],
        title: Optional[str] = None,
        colors: Union[str, List[str]] = 'turquoise',
        labels: Optional[Union[str, List[str]]] = None,
        ylim: Optional[tuple] = None,
        ax: Optional[plt.Axes] = None
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
    ax.set_xlim(-0.02, 1.02)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(r'Significance level, $\alpha$')
    ax.set_ylabel('Test power')

    return ax


def compute_auc(sig_levels: np.ndarray, power: np.ndarray):
    return np.trapz(power, x=sig_levels)  # could also use scipy.integrate.simpson or sklearn.metrics.auc


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
