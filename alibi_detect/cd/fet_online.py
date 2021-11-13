from tqdm import tqdm
import numpy as np
from typing import Any, Callable, List, Optional, Union
from alibi_detect.cd.base_online import BaseUniDriftOnline
from alibi_detect.utils.misc import quantile
from alibi_detect.cd.fet import fisher_exact
import numba as nb
import warnings


class FETDriftOnline(BaseUniDriftOnline):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_sizes: List[int],
            preprocess_fn: Optional[Callable] = None,
            n_bootstraps: int = 10000,
            alternative: str = 'less',
            lam: float = 0.99,
            t_max: Optional[int] = None,
            n_features: Optional[int] = None,
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
        window_sizes
            window sizes for the sliding test-windows used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ERT.
        alternative
            Defines the alternative hypothesis. Options are 'less', 'greater' or 'two-sided'.
        lam
            Smoothing coefficient used for exponential moving average.
        t_max
            Length of streams to simulate. If `None`, this is set to 2 * max(`window_sizes`) - 1.
        n_features
            Number of features used in the statistical test. No need to pass it if no preprocessing takes place.
            In case of a preprocessing step, this can also be inferred automatically but could be more
            expensive to compute.
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
            window_sizes=window_sizes,
            preprocess_fn=preprocess_fn,
            n_bootstraps=n_bootstraps,
            n_features=n_features,
            verbose=verbose,
            input_shape=input_shape,
            data_type=data_type
        )
        self.lam = lam
        if alternative.lower() not in ['less', 'greater', 'two-sided']:
            raise ValueError("`alternative` must be either 'less', 'greater' or 'two-sided'.")
        self.alternative = alternative.lower()

        # Stream length
        if t_max is not None:
            if t_max < 2 * self.max_ws - 1:
                raise ValueError("`t_max` must be >= 2 * max(`window_sizes`) for the FETDriftOnline detector.")
        else:
            t_max = 2 * self.max_ws - 1
        self.t_max = t_max

        # Check data is only [False, True] or [0, 1]
        values = set(np.unique(x_ref))
        if values != {True, False} and values != {0, 1}:
            raise ValueError("The `x_ref` data must consist of only (0,1)'s or (False,True)'s for the "
                             "FETDriftOnline detector.")

        # Configure thresholds and initialise detector
        self._initialise()
        self._configure_thresholds()

    def _configure_ref(self) -> None:
        self.sum_ref = np.sum(self.x_ref, axis=0)

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

        # Assuming independent features, calibrate to beta = 1 - (1-FPR)^(1/n_features)
        beta = 1 - (1-self.fpr)**(1/self.n_features)

        # Compute test statistic at each t_max number of t's, for each stream and each feature
        thresholds = np.zeros((self.t_max, self.n_features), dtype=np.float32)
        self.permit_probs = np.full((self.t_max, self.n_features), np.nan)
        if self.verbose:
            pbar = tqdm(total=int(self.n_features*len(self.window_sizes)),
                        desc="Simulating %d streams for %d features" % (self.n_bootstraps, self.n_features))
        else:
            pbar = None
        for f in range(self.n_features):
            # Compute stats for given feature (for each stream)
            stats = self._simulate_streams(self.x_ref[:, f], pbar)
            # At each t for each stream, find max stats. over window sizes
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
                max_stats = np.nanmax(stats, -1)
            # Find threshold (at each t) that satisfies eqn. (2) in Ross et al.
            for t in range(self.t_max):
                if t < np.min(self.window_sizes):
                    thresholds[t, f] = np.nan
                else:
                    # Compute (1-beta) quantile of max_stats at a given t, over all streams
                    threshold = np.float32(quantile(max_stats[:, t], 1 - beta, interpolate=False, type=6))
                    stats_below = max_stats[max_stats[:, t] < threshold]
                    # Check for stats equal to threshold
                    prob_of_equal = (max_stats[:, t] <= threshold).mean() - (max_stats[:, t] < threshold).mean()
                    if prob_of_equal == 0.0:
                        permit_prob = np.inf
                        max_stats = stats_below  # Remove streams where change point detected
                    else:
                        undershoot = 1 - beta - (max_stats[:, t] < threshold).mean()
                        permit_prob = undershoot / prob_of_equal
                        stats_equal = max_stats[max_stats[:, t] == threshold]
                        n_keep_equal = np.random.binomial(len(stats_equal), permit_prob)
                        # Remove streams where change point detected, but allow permit_prob streams where stats=thresh
                        max_stats = np.concatenate([stats_below, stats_equal[:n_keep_equal]])

                    thresholds[t, f] = threshold
                    self.permit_probs[t, f] = permit_prob
        self.thresholds = thresholds

    def _simulate_streams(self, x_ref: np.ndarray, pbar: Optional[tqdm]) -> np.ndarray:
        """
        Computes test statistic for each stream.

        Almost all of the work done here is done in a call to scipy's hypergeom for each window size.
        """
        n_windows = len(self.window_sizes)
        stats = np.full((self.n_bootstraps, self.t_max, n_windows), np.nan)

        p = np.mean(x_ref)
        sum_ref = np.sum(x_ref)
        x_stream = np.random.choice([False, True], (self.n_bootstraps, self.t_max), p=[p, 1 - p])
        cumsums_stream = np.cumsum(x_stream, axis=-1)
        for k in range(n_windows):
            if pbar is not None:
                pbar.update(1)
            ws = self.window_sizes[k]
            cumsums_last_ws = cumsums_stream[:, ws:] - cumsums_stream[:, :-ws]

            # Construct contingency table and perform fisher's exact test
            a = cumsums_last_ws
            b = np.full_like(a, sum_ref)
            c = ws - cumsums_last_ws
            d = np.full_like(c, self.n - sum_ref)
            _, p_val = fisher_exact(a, b, c, d, alternative=self.alternative)

            stats[:, ws:, k] = self._exp_moving_avg(1 - p_val, self.lam)
        return stats

    @staticmethod
    @nb.njit(cache=True)
    def _exp_moving_avg(arr: np.ndarray, lam: float) -> np.ndarray:
        """ Apply exponential moving average over the final axis."""
        output = np.zeros_like(arr)
        output[..., 0] = arr[..., 0]
        for i in range(1, arr.shape[-1]):
            output[..., i] = (1 - lam) * output[..., i - 1] + lam * arr[..., i]
        return output

    def _update_state(self, x_t: np.ndarray):
        self.t += 1
        if self.t == 1:
            # Initialise stream
            self.xs = x_t
        else:
            # Update stream
            self.xs = np.concatenate([self.xs, x_t])

    def _check_drift(self, test_stats: np.ndarray, thresholds: np.ndarray) -> int:
        """
        Private method to compare test stats to thresholds. The max stats over all windows are compute for each
        feature. Drift is flagged if `max_stats` for any feature exceeds the thresholds for that feature.

        Parameters
        ----------
        test_stats
            Array of test statistics with shape (n_windows, n_features)
        thresholds
            Array of thresholds with shape (t_max, n_features).

        Returns
        -------
        An int equal to 1 if drift, 0 otherwise.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
            max_stats = np.nanmax(test_stats, axis=0)

        # If any stats greater than thresholds, flag drift and return
        if (max_stats > thresholds).any():
            return 1

        # If still no drift, check if any stats equal to threshold. If so, flag drift with proba self.probs_when_equal
        equal_inds = np.where(max_stats == thresholds)[0]
        for equal_ind in equal_inds:
            if np.random.uniform() > self.permit_probs[min(self.t-1, len(self.thresholds)-1), equal_ind]:
                return 1

        return 0

    def score(self, x_t: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Compute the test-statistic (FET) between the reference window(s) and test window.
        If a given test-window is not yet full then a test-statistic of np.nan is returned for that window.

        Parameters
        ----------
        x_t
            A single instance.

        Returns
        -------
        Estimated FET test statistics (1-p_val) between reference window and test windows.
        """
        x_t = super()._preprocess_xt(x_t)
        self._update_state(x_t)

        stats = np.zeros((len(self.window_sizes), self.n_features), dtype=np.float32)
        for k, ws in enumerate(self.window_sizes):
            if self.t >= ws:
                sum_last_ws = np.sum(self.xs[-ws:, :], axis=0)
                # Construct contingency table and perform fisher's exact test
                a = sum_last_ws
                b = np.full_like(a, self.sum_ref)
                c = ws - sum_last_ws
                d = np.full_like(c, self.n - self.sum_ref)
                _, p_val = fisher_exact(a, b, c, d, alternative=self.alternative)

                # Compute test stat and apply smoothing
                stat = 1 - p_val
                if len(self.test_stats) != 0:
                    tmp_stats = np.nan_to_num(self.test_stats[-1, k], nan=0.0)
                    stat = (1-self.lam)*tmp_stats + self.lam*stat
                stats[k] = stat
            else:
                stats[k] = np.nan
        return stats
