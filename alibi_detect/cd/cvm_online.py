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
            window_size: Union[int, List[int]],  # TODO - this type hint conflicts with parent class
            preprocess_fn: Optional[Callable] = None,
            n_bootstraps: int = 10000,
            device='cpu',
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

        # Preprocess reference data
        if isinstance(self.x_ref, List):
            self.x_ref = np.array(x_ref)
        self.x_ref = self.x_ref.squeeze()  # squeeze in case of (n,1) array
        if self.x_ref.ndim != 1:
            raise ValueError("The `x_ref` data must be 1D for the CVMDriftOnline detector.")

        # Get max and min window sizes
        self.window_size = np.array(self.window_size)
        self.max_ws = np.max(self.window_size)
        self.min_ws = np.min(self.window_size)

        # Configure thresholds and initialise detector
        self._configure_thresholds()
        self._initialise()

    def _configure_ref(self):
        ids_ref_ref = self.x_ref[None, :] >= self.x_ref[:, None]
        self.ref_cdf_ref = np.sum(ids_ref_ref, axis=0)/self.n

    def _configure_thresholds(self):
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

        # Compute test statistic at each T number of t's, for each of the n_bootstrap number of streams
        T = 2 * np.max(self.window_size) - 1  # Should be constant after T*max_ws-1. #TODO - option to set manually?
        stats = self._compute_stats(self.n, T, self.n_bootstraps, self.window_size)
        # At each t for each stream, find max stats. over window sizes
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
            max_stats = np.nanmax(stats, -1)  # TODO - get rid of "RuntimeWarning: All-NaN slice encountered" here
        # Now loop through each t and find threshold (at each t) that satisfies eqn. (2) in Ross et al.
        thresholds = np.zeros(T)
        for t in range(T):
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
    def _compute_stats(
            N: int, T: int, B: int, window_sizes: np.ndarray, zs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Important that we guvectorise over the parallel streams. Not sufficient just to write a
        normal vectorised numpy implementation as this can lead to OOM errors (when trying to store
        (N+T) x (N+T) x B matrices of floats). However we will store the boolean matrix of this size as
        it faster to compute this way (and 64x smaller).
        """
        K = len(window_sizes)
        zs = np.random.randn(B, N + T) if zs is None else zs
        ids = zs[:, None, :] >= zs[:, :, None]
        # guvectorise from here to vectorise over first dim of ids
        # works by filling in the entries of the last arg it is passed
        stats = np.zeros((B, T, K))
        _ids_to_stats(ids[:, :N, :], ids[:, N:, :], window_sizes, stats)
        # Remove stats prior to windows being full
        for k, ws in enumerate(window_sizes):
            stats[:ws, k] = np.nan
        return stats

    def score(self, x_t: np.ndarray) -> Union[float, None]:
        """
        Compute the test-statistic (squared MMD) between the reference window and test window.
        If the test-window is not yet full then a test-statistic of None is returned.

        Parameters
        ----------
        x_t
            A single instance.

        Returns
        -------
        Squared MMD estimate between reference window and test window.
        """
        # TODO
        pass

    # TODO - Override predict from base_online.py


@nb.njit
def _normalise_stats(stats: np.ndarray, N: int, ws: int) -> np.ndarray:
    """
    See Eqns 3 & 14 of https://www.projecteuclid.org/euclid.aoms/1177704477.
    """
    mu = 1 / 6 + 1 / (6 * (N + ws))
    var_num = (N + ws + 1) * (4 * N * ws * (N + ws) - 3 * (N * N + ws * ws) - 2 * N * ws)
    var_denom = 45 * (N + ws) * (N + ws) * 4 * N * ws
    prod = N * ws / ((N + ws) * (N + ws))
    return (stats * prod - mu) / np.sqrt(var_num / var_denom)


@guvectorize(
    [(nb.boolean[:, :], nb.boolean[:, :], nb.int64[:], nb.float64[:, :])],
    '(N,N_ALL), (T,N_ALL), (K) -> (T,K)',
    nopython=True,
    target="parallel"   # TODO - Always parallel? or switch to cpu for small data and cuda for big?
                        # TODO - cuda>parallel>cuda for speed, but opposite for overhead
                        # TODO - Need to set this from self.device
)
def _ids_to_stats(
        ids_ref_all: np.ndarray,
        ids_stream_all: np.ndarray,
        window_sizes: np.ndarray,
        stats: np.ndarray,
) -> np.ndarray:
    N = ids_ref_all.shape[0]
    T = ids_stream_all.shape[0]
    N_ALL = ids_ref_all.shape[-1]
    K = window_sizes.shape[0]
    ref_cdf_all = np.sum(ids_ref_all, axis=0) / N

    cumsums = np.zeros((T, N_ALL))   # TODO - how to get progress bar?
    for i in range(N_ALL):
        cumsums[:, i] = np.cumsum(ids_stream_all[:, i])

    for k in range(K):
        ws = window_sizes[k]
        win_cdf_ref = (cumsums[ws:, :N] - cumsums[:-ws, :N]) / ws
        cdf_diffs_on_ref = np.expand_dims(ref_cdf_all[:N], 0) - win_cdf_ref
        stats[ws:, k] = np.sum(cdf_diffs_on_ref * cdf_diffs_on_ref, axis=-1)
        for t in range(ws, T):
            win_cdf_win = (cumsums[t, N + t - ws:N + t] - cumsums[t - ws, N + t - ws:N + t]) / ws
            cdf_diffs_on_win = np.expand_dims(ref_cdf_all[N + t - ws:N + t], 0) - win_cdf_win
            stats[t, k] += np.sum(cdf_diffs_on_win * cdf_diffs_on_win)
        stats[:, k] = _normalise_stats(stats[:, k], N, ws)
