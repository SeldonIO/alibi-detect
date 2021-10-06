import numpy as np
from typing import Callable, List, Optional, Union
from alibi_detect.cd.base_online import BaseDriftOnline
import numba as nb
from numba.np.ufunc.decorators import guvectorize
import warnings


class CVMDriftOnline(BaseDriftOnline):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: Union[int, List[int]],
            preprocess_fn: Optional[Callable] = None,
            n_bootstraps: int = 10000,
            device='parallel',
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Online Cramer Von-Mises (CVM) data drift detector using preconfigured thresholds.

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
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. TODO: decide on default
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

        self.device = device

        if isinstance(window_size, int):
            self.window_size = [window_size]  # type: list

        # Preprocess reference data
        self.x_ref = self.x_ref.squeeze()  # squeeze in case of (n,1) array
        if self.x_ref.ndim != 1:
            raise ValueError("The `x_ref` data must be 1D for the CVMDriftOnline detector.")

        # Get max and min window sizes
        self.max_ws = np.max(self.window_size)
        self.min_ws = np.min(self.window_size)

        # Configure thresholds and initialise detector
        self._configure_thresholds()
        self._initialise()

    def _configure_ref(self) -> None:
        ids_ref_ref = self.x_ref[None, :] >= self.x_ref[:, None]
        self.ref_cdf_ref = np.sum(ids_ref_ref, axis=0)/self.n

    def _configure_thresholds(self) -> None:
        """
        Private method to simulate trajectories of the Cramer Von-Mises statistic for the desired reference set
        size and window sizes under the null distribution, where both the reference set and deployment stream
        follow the same distribution. It then uses these simulated trajectories to estimate thresholds.

        As the test statistics are rank based and independent of the underlying distribution, we may use any
        continuous distribution -- we use Gaussian.

        The thresholds should stop changing after t=(2*max-window-size - 1) and therefore we need only simulate
        trajectories and estimate thresholds up to this point.
        """
        if self.verbose:
            print("Using %d boostrap simulations to configure thresholds..." % self.n_bootstraps)

        # Compute test statistic at each t_max number of t's, for each of the n_bootstrap number of streams
        t_max = 2 * np.max(self.window_size) - 1  # Should be constant after t_max*max_ws-1.
        stats = self._simulate_streams(self.n, t_max, self.n_bootstraps, self.window_size)
        # At each t for each stream, find max stats. over window sizes
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
            max_stats = np.nanmax(stats, -1)
        # Now loop through each t and find threshold (at each t) that satisfies eqn. (2) in Ross et al.
        thresholds = np.zeros(t_max)
        for t in range(t_max):
            if t < np.min(self.window_size):
                thresholds[t] = np.nan  # Set to NaN prior to window being full
            else:
                # TODO - Next two lines need more careful thought
                # Compute (1-fpr) quantile of max_stats at a given t, over all streams
                threshold = np.quantile(max_stats[:, t], 1 - self.fpr)
                # Remove streams for which a change point has already been detected
                max_stats = max_stats[max_stats[:, t] <= threshold]
                thresholds[t] = threshold

        self.thresholds = thresholds

    @staticmethod
    def _simulate_streams(
            n: int, t_max: int, n_bootstraps: int, window_sizes: list, xs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Important that we guvectorise over the parallel streams. Not sufficient just to write a
        normal vectorised numpy implementation as this can lead to OOM errors (when trying to store
        (n+t_max) x (n+t_max) x n_bootstraps matrices of floats). However we will store the boolean
        matrix of this size as it faster to compute this way (and 64x smaller).
        """
        n_windows = len(window_sizes)
        xs = np.random.randn(n_bootstraps, n + t_max) if xs is None else xs
        ids = xs[:, None, :] >= xs[:, :, None]
        # guvectorise from here to vectorise over first dim of ids
        # works by filling in the entries of the last arg it is passed
        stats = np.zeros((n_bootstraps, t_max, n_windows))
        # TODO - Allows us to set self.device at runtime, but a little slower. cache=True speeds up subsequent runs.
#        _ids_to_stats_nb = guvectorize(
#                                    [(nb.boolean[:, :], nb.boolean[:, :], nb.int64[:], nb.float64[:, :])],
#                                    '(n,n_all), (t_max,n_all), (n_windows) -> (t_max,n_windows)',
#                                    nopython=True, target=self.device, cache=True,
#                                    )(_ids_to_stats)
#        _ids_to_stats_nb(ids[:, :n, :], ids[:, n:, :], window_sizes, stats)
        _ids_to_stats(ids[:, :n, :], ids[:, n:, :], np.asarray(window_sizes), stats)
        # Remove stats prior to windows being full
        for k, ws in enumerate(window_sizes):
            stats[:ws, k] = np.nan
        return stats

    def _update_state(self, x_t: Optional[np.ndarray] = None) -> None:
        # if self.t == 1:  # TODO - this doesn't work if score() called before predict(), as self.t==0. Hence do below.
        if not hasattr(self, 'xs'):
            # Initialise stream
            self.xs = x_t
            self.ids_ref_wins = (x_t >= self.x_ref)[:, None]
            self.ids_wins_ref = (x_t <= self.x_ref)[None, :]
            self.ids_wins_wins = np.array(1).reshape(1, 1)
        else:
            # Update stream
            self.xs = np.concatenate([self.xs, x_t])
            self.ids_ref_wins = np.concatenate(
                [self.ids_ref_wins[:, -(self.max_ws-1):], (x_t >= self.x_ref)[:, None]], 1
            )
            self.ids_wins_ref = np.concatenate(
                [self.ids_wins_ref[-(self.max_ws-1):], (x_t <= self.x_ref)[None, :]], 0
            )
            self.ids_wins_wins = np.concatenate(
                [self.ids_wins_wins[-(self.max_ws-1):, -(self.max_ws-1):],
                 (x_t >= self.xs[-self.max_ws:-1])[:, None]], 1
            )
            self.ids_wins_wins = np.concatenate(
                [self.ids_wins_wins, (x_t <= self.xs[-self.max_ws:])[None, :]], 0
            )

    def score(self, x_t: np.ndarray) -> np.ndarray:
        """
        Compute the test-statistic (CVM) between the reference window(s) and test window.
        If a given test-window is not yet full then a test-statistic of NaN is returned for that window.

        Parameters
        ----------
        x_t
            A single instance.

        Returns
        -------
        Estimated CVM test statistics between reference window and test window(s).
        """
        if isinstance(x_t, int) or isinstance(x_t, float):  # we expect ndarray but convert these for convenience
            x_t = np.array([x_t])
        if x_t.ndim != 1:
            raise ValueError("The `x_t` passed to score() data must be 1D ndarray of length 1.")
        self._update_state(x_t)

        stats = np.zeros_like(self.window_size, dtype=np.float32)
        for k, ws in enumerate(self.window_size):
            if self.t >= ws:
                ref_cdf_win = np.sum(self.ids_ref_wins[:, -ws:], axis=0)/self.n
                win_cdf_ref = np.sum(self.ids_wins_ref[-ws:], axis=0)/ws
                win_cdf_win = np.sum(self.ids_wins_wins[-ws:, -ws:], axis=0)/ws
                ref_cdf_diffs = self.ref_cdf_ref - win_cdf_ref
                win_cdf_diffs = ref_cdf_win - win_cdf_win
                sum_diffs_2 = np.sum(ref_cdf_diffs*ref_cdf_diffs) + np.sum(win_cdf_diffs*win_cdf_diffs)
                stats[k] = _normalise_stats(sum_diffs_2, self.n, ws)
            else:
                stats[k] = np.nan
        return stats


@nb.njit
def _normalise_stats(stats: np.ndarray, n: int, ws: int) -> np.ndarray:
    """
    See Eqns 3 & 14 of https://www.projecteuclid.org/euclid.aoms/1177704477.
    """
    mu = 1 / 6 + 1 / (6 * (n + ws))
    var_num = (n + ws + 1) * (4 * n * ws * (n + ws) - 3 * (n * n + ws * ws) - 2 * n * ws)
    var_denom = 45 * (n + ws) * (n + ws) * 4 * n * ws
    prod = n * ws / ((n + ws) * (n + ws))
    return (stats * prod - mu) / np.sqrt(var_num / var_denom)


@guvectorize(
    [(nb.boolean[:, :], nb.boolean[:, :], nb.int64[:], nb.float64[:, :])],
    '(n,n_all), (t_max,n_all), (n_windows) -> (t_max,n_windows)',
    nopython=True,
    target="parallel"  # TODO - Always parallel? or switch to cpu for small data and cuda for big?
    # TODO - cuda>parallel>cuda for speed, but opposite for overhead
    # TODO - Need to set this from self.device (would mean putting back in as class method, which adds runtime ops)
)
def _ids_to_stats(
        ids_ref_all: np.ndarray,
        ids_stream_all: np.ndarray,
        window_sizes: np.ndarray,
        stats: np.ndarray,
) -> np.ndarray:  # type: ignore
    n = ids_ref_all.shape[0]
    t_max = ids_stream_all.shape[0]
    n_all = ids_ref_all.shape[-1]
    n_windows = window_sizes.shape[0]
    ref_cdf_all = np.sum(ids_ref_all, axis=0) / n

    cumsums = np.zeros((t_max, n_all))  # TODO - how to get progress bar?
    for i in range(n_all):
        cumsums[:, i] = np.cumsum(ids_stream_all[:, i])

    for k in range(n_windows):
        ws = window_sizes[k]
        win_cdf_ref = (cumsums[ws:, :n] - cumsums[:-ws, :n]) / ws
        cdf_diffs_on_ref = np.expand_dims(ref_cdf_all[:n], 0) - win_cdf_ref
        stats[ws:, k] = np.sum(cdf_diffs_on_ref * cdf_diffs_on_ref, axis=-1)
        for t in range(ws, t_max):
            win_cdf_win = (cumsums[t, n + t - ws:n + t] - cumsums[t - ws, n + t - ws:n + t]) / ws
            cdf_diffs_on_win = np.expand_dims(ref_cdf_all[n + t - ws:n + t], 0) - win_cdf_win
            stats[t, k] += np.sum(cdf_diffs_on_win * cdf_diffs_on_win)
        stats[:, k] = _normalise_stats(stats[:, k], n, ws)
