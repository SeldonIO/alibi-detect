from tqdm import tqdm
import numpy as np
from scipy.stats import hypergeom
from typing import Callable, List, Optional, Union
from alibi_detect.cd.base_online import BaseDriftOnline
from alibi_detect.utils.misc import quantile
from numba import njit
import warnings


class FETDriftOnline(BaseDriftOnline):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: Union[int, List[int]],
            preprocess_fn: Optional[Callable] = None,
            n_bootstraps: int = 10000,
            alternative: str = 'less',
            lam: float = 0.99,
            t_max: Optional[int] = None,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Online Fisher exact test (FET) data drift detector using preconfigured thresholds.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift.
        window_size
            window size(s) for the sliding test-window(s) used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ERT.
        alternative
            Defines the alternative hypothesis. Options are 'less' or 'greater'.
        lam
            Smoothing coefficient used for exponential moving average.
        t_max
            Length of streams to simulate. If `None`, this is set to 2 * max(`window_size`) - 1.
        verbose
            Whether or not to print progress during configuration.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            ert=ert,
            window_size=window_size,
            preprocess_fn=preprocess_fn,
            n_bootstraps=n_bootstraps,
            verbose=verbose,
            input_shape=input_shape,
            data_type=data_type
        )
        self.lam = lam
        if alternative.lower() not in ['less', 'greater']:
            raise ValueError("`alternative` must be either 'less' or 'greater'.")
        self.alternative = alternative.lower()

        # Window sizes
        if isinstance(window_size, int):
            self.window_size: list = [window_size]
        self.max_ws = np.max(self.window_size)
        self.min_ws = np.min(self.window_size)

        # Stream length
        if t_max is not None:
            if t_max < 2 * self.max_ws - 1:
                raise ValueError("`t_max` must be >= 2 * max(`window_size`) for the FETDriftOnline detector.")
        self.t_max = t_max

        # Preprocess reference data
        self.x_ref = self.x_ref.squeeze()  # squeeze in case of (n,1) array
        if self.x_ref.ndim != 1:
            raise ValueError("The `x_ref` data must be 1D for the FETDriftOnline detector.")

        # Check data is only [False, True] or [0, 1]
        values = set(np.unique(x_ref))
        if values != {True, False} and values != {0, 1}:
            raise ValueError("The `x_ref` data must consist of only [0,1]'s or [False,True]'s for the "
                             "FETDriftOnline detector.")

        # Configure thresholds and initialise detector
        self._configure_thresholds()
        self._initialise()

    def _configure_ref(self) -> None:
        self.sum_ref = np.sum(self.x_ref)

    def _configure_thresholds(self) -> None:
        """
        A function that simulates trajectories of the (smoothed) Fisher exact test statistic for the desired
        reference set and window sizes under the null distribution, where both the reference set and deployment
        stream follow the same distribution. It then uses these simulated trajectories to estimate thresholds.

        The test statistics are smoothed using an exponential moving average to remove their discreteness and
        therefore allow more precise quantiles to be targeted.

        The thresholds should stop changing after t=(2*max-window-size - 1) and therefore we need only simulate
        trajectories and estimate thresholds up to this point.
        """
        if self.verbose:
            print("Using %d bootstrap simulations to configure thresholds..." % self.n_bootstraps)

        # Compute test statistic at each t_max number of t's, for each of the n_bootstrap number of streams
        t_max = 2 * self.max_ws - 1 if self.t_max is None else self.t_max
        stats = self._simulate_streams(self.n, t_max, self.n_bootstraps, self.window_size,
                                       float(np.mean(self.x_ref)), self.lam)
        # At each t for each stream, find max stats. over window sizes
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
            max_stats = np.nanmax(stats, -1)
        # Now loop through each t and find threshold (at each t) that satisfies eqn. (2) in Ross et al.
        thresholds = np.zeros(t_max)
        for t in range(t_max):
            if t < np.min(self.window_size):
                thresholds[t] = np.nan
            else:
                # Compute (1-fpr) quantile of max_stats at a given t, over all streams
                threshold = quantile(max_stats[:, t], 1 - self.fpr)
                # Remove streams for which a change point has already been detected
                max_stats = max_stats[max_stats[:, t] <= threshold]
                thresholds[t] = threshold
        self.thresholds = thresholds

    def _simulate_streams(
            self, n: int, t_max: int, n_bootstraps: int, window_sizes: list, p: float, lam: float
    ) -> np.ndarray:
        """
        Computes test statistic for each stream.

        Almost all of the work done here is done in a call to scipy's hypergeom for each window size.
        """
        n_windows = len(window_sizes)
        stats = np.full((n_bootstraps, t_max, n_windows), np.nan)

        z_ref = np.random.choice([False, True], (n_bootstraps, n), p=[p, 1 - p])
        z_stream = np.random.choice([False, True], (n_bootstraps, t_max), p=[p, 1 - p])
        sum_ref = np.sum(z_ref, axis=-1)
        cumsums_stream = np.cumsum(z_stream, axis=-1)
        for k in tqdm(range(n_windows)):
            ws = window_sizes[k]
            cumsums_last_ws = cumsums_stream[:, ws:] - cumsums_stream[:, :-ws]

            if self.alternative == 'less':
                p_val = hypergeom.cdf(sum_ref[:, None], n+ws, sum_ref[:, None] + cumsums_last_ws, n)
            elif self.alternative == 'greater':
                p_val = hypergeom.cdf(cumsums_last_ws, n+ws, sum_ref[:, None] + cumsums_last_ws, ws)
            else:
                raise ValueError("'alternative' not yet implemented.")
            stats[:, ws:, k] = self._exp_moving_avg(1 - p_val, lam)
        return stats

    @staticmethod
    @njit
    def _exp_moving_avg(arr: np.ndarray, lam: float) -> np.ndarray:
        """ Apply exponential moving average over the final axis."""
        output = np.zeros_like(arr)
        output[..., 0] = arr[..., 0]
        for i in range(1, arr.shape[-1]):
            output[..., i] = (1 - lam) * output[..., i - 1] + lam * arr[..., i]
        return output

    def _update_state(self, x_t: np.ndarray):
        self.t += 1

        # Preprocess x_t
        x_t = super()._preprocess_xt(x_t)
        if x_t.ndim != 1:
            raise ValueError("The `x_t` passed to score() data must be 1D ndarray of length 1.")

        # Init or update state
        if self.t == 1:
            # Initialise stream
            self.xs = x_t
        else:
            # Update stream
            self.xs = np.concatenate([self.xs, x_t])

    def score(self, x_t: np.ndarray) -> np.ndarray:
        """
        Compute the test-statistic (FET) between the reference window(s) and test window.
        If a given test-window is not yet full then a test-statistic of np.nan is returned for that window.

        Parameters
        ----------
        x_t
            A single instance.

        Returns
        -------
        Estimated FET test statistics between reference window and test window(s).
        """
        self._update_state(x_t)

        stats = np.zeros_like(self.window_size, dtype=np.float32)
        for k, ws in enumerate(self.window_size):
            if self.t >= ws:
                sum_last_ws = np.sum(self.xs[-ws:])
                if self.alternative == 'less':
                    p_val = hypergeom.cdf(self.sum_ref, self.n+ws, self.sum_ref + sum_last_ws, self.n)
                else:
                    p_val = hypergeom.cdf(sum_last_ws, self.n+ws, self.sum_ref + sum_last_ws, ws)

                stat = 1 - p_val
                if len(self.test_stats) != 0 and not np.isnan(self.test_stats[-1, k]):
                    stat = (1-self.lam)*self.test_stats[-1, k] + self.lam*stat
                stats[k] = stat
            else:
                stats[k] = np.nan
        return stats
