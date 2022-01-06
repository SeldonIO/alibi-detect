import logging
import numpy as np
from typing import Dict
from alibi_detect.base import BaseDetector, ThresholdMixin, outlier_prediction_dict

logger = logging.getLogger(__name__)

# numerical stability for division
EPSILON = 1e-8

# padding values
PADDING_CONSTANT = 'constant'
PADDING_REPLICATE = 'replicate'
PADDING_REFLECT = 'reflect'
PADDINGS = [PADDING_CONSTANT, PADDING_REFLECT, PADDING_REPLICATE]

# padding sides
SIDE_BILATERAL = 'bilateral'
SIDE_LEFT = 'left'
SIDE_RIGHT = 'right'
SIDES = [SIDE_BILATERAL, SIDE_RIGHT, SIDE_LEFT]


class SpectralResidual(BaseDetector, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 window_amp: int = None,
                 window_local: int = None,
                 padding_amp_method: str = PADDING_REFLECT,
                 padding_local_method: str = PADDING_REFLECT,
                 padding_amp_side: str = SIDE_BILATERAL,
                 n_est_points: int = None,
                 n_grad_points: int = 5,
                 ) -> None:
        """
        Outlier detector for time-series data using the spectral residual algorithm.
        Based on "Time-Series Anomaly Detection Service at Microsoft" (Ren et al., 2019)
        https://arxiv.org/abs/1906.03821

        Parameters
        ----------
        threshold
            Threshold used to classify outliers. Relative saliency map distance from the moving average.
        window_amp
            Window for the average log amplitude.
        window_local
            Window for the local average of the saliency map. Note that the averaging is performed over the
            previous `window_local` data points (i.e., is a local average of the preceding `window_local` points for
            the current index).
        padding_amp_method
            Padding method to be used prior to each convolution over log amplitude.
            Possible values: `constant` | `replicate` | `reflect`. Default value: `replicate`.
             - `constant` - padding with constant `0`.
             - `replicate` - repeats the last/extreme value.
             - `reflect` - reflects the signal.
        padding_local_method
            Padding method to be used prior to each convolution over saliency map.
            Possible values: `constant` | `replicate` | `reflect`. Default value: `replicate`.
             - `constant` - padding with constant `0`.
             - `replicate` - repeats the last/extreme value.
             - `reflect` - reflects the signal.
        padding_amp_side
            Whether to pad the amplitudes on both sides or only on one side.
            Possible values: `bilateral` | `left` | `right`.
        n_est_points
            Number of estimated points padded to the end of the sequence.
        n_grad_points
            Number of points used for the gradient estimation of the additional points padded
            to the end of the sequence.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.window_amp = window_amp
        self.window_local = window_local
        self.conv_amp = np.ones((1, window_amp)).reshape(-1,) / window_amp

        # conv_local needs a special treatment since the paper says that:
        # \bar{S}(xi) is the local average of the preceding z points of S(xi).
        # To use the same padding implementation that includes the current point we convolving, we define a modified
        # filter given by: [0, 1, 1, 1,  ... ,1] / window_local of size `window_local + 1`. In this way
        # the current value is multiplied by 0 and thus neglected. Note that the 0 goes first since before the
        # element-wise multiplication, the filter is flipped. We only do this since the filter is asymmetric.
        self.conv_local = np.ones((1, window_local + 1)).reshape(-1,) / window_local
        self.conv_local[0] = 0

        self.n_est_points = n_est_points
        self.n_grad_points = n_grad_points
        self.padding_amp_method = padding_amp_method
        self.padding_local_method = padding_local_method
        self.padding_amp_side = padding_amp_side

        # set metadata
        self.meta['detector_type'] = 'online'
        self.meta['data_type'] = 'time-series'

    def infer_threshold(self,
                        X: np.ndarray,
                        t: np.ndarray = None,
                        threshold_perc: float = 95.
                        ) -> None:
        """
        Update threshold by a value inferred from the percentage of instances considered to be
        outliers in a sample of the dataset.

        Parameters
        ----------
        X
            Batch of instances.
        threshold_perc
            Percentage of X considered to be normal based on the outlier score.
        """
        if t is None:
            t = np.arange(X.shape[0])

        # compute outlier scores
        iscore = self.score(X, t)

        # update threshold
        self.threshold = np.percentile(iscore, threshold_perc)

    @staticmethod
    def pad_same(X: np.ndarray,
                 W: np.ndarray,
                 method: str = "replicate",
                 side: str = 'bilateral'
                 ) -> np.ndarray:
        """
        Adds padding to the signal `X` such that after applying a valid convolution with a kernel/filter
        `w`, the resulting signal has the same shape as the input `X`.

        Parameters
        ----------
        X
            Signal to be padded
        W
            Convolution kernel/filter.
        method
            Padding method to be used.
            Possible values:
             - `constant` - padding with constant `0`.
             - `replicate` - repeats the last/extreme value.
             - `reflect` - reflects the signal.
        side
            Whether to pad the signal bilateral or only on one side.
            Possible values:
             - `bilateral` - signal is padded on both sides
             - `left` - signal is padded only on the left hand side.
             - `right` - signal is padded only on the right hand side.

        Returns
        -------
        Padded signal.
        """
        if method not in PADDINGS:
            raise ValueError(f'Unknown padding method. Received {method}. Select one of the following: {PADDINGS}')

        if side not in SIDES:
            raise ValueError(f'Unknown padding side. Receive {side}. Select one of the follwoing: {SIDES}')

        assert len(X.shape) == 1, "Only 1D signal supported."
        assert len(W.shape) == 1, "Only 1D kernel/filter supported."
        pad_size = W.shape[0] - 1

        if side == SIDE_BILATERAL:
            pad_size_right = pad_size // 2
            pad_size_left = pad_size - pad_size_right

        elif side == SIDE_LEFT:
            pad_size_right = 0
            pad_size_left = pad_size

        else:
            pad_size_right = pad_size
            pad_size_left = 0

        # replicate padding
        if method == PADDING_REPLICATE:
            return np.concatenate([
                np.tile(X[0], pad_size_left),
                X,
                np.tile(X[-1], pad_size_right)
            ])

        # reflection padding
        if method == PADDING_REFLECT:
            return np.concatenate([
                X[1:pad_size_left + 1][::-1],
                X,
                X[-pad_size_right - 1: -1][::-1] if pad_size_right > 0 else np.array([])
            ])

        # zero padding
        return np.concatenate([
            np.tile(0, pad_size_left),
            X,
            np.tile(0, pad_size_right)
        ])

    def saliency_map(self, X: np.ndarray) -> np.ndarray:
        """
        Compute saliency map.

        Parameters
        ----------
        X
            Time series of instances.

        Returns
        -------
        Array with saliency map values.
        """
        assert X.shape[0] > self.conv_amp.shape[0], "The length of the input signal should be greater " \
                                                    "than the amplitude window"

        fft = np.fft.fft(X)
        amp = np.abs(fft)
        log_amp = np.log(amp)
        phase = np.angle(fft)
        # split spectrum into bias term and symmetric frequencies
        bias, sym_freq = log_amp[:1], log_amp[1:]
        # select just the first half of the sym_freq
        freq = sym_freq[:(len(sym_freq) + 1) // 2]
        # apply filter/moving average, but first pad the `freq` array
        padded_freq = SpectralResidual.pad_same(X=freq,
                                                W=self.conv_amp,
                                                method=self.padding_amp_method,
                                                side=self.padding_amp_side)
        ma_freq = np.convolve(padded_freq, self.conv_amp, 'valid')
        # construct moving average log amplitude spectrum
        ma_log_amp = np.concatenate([
            bias,
            ma_freq,
            (ma_freq[:-1] if len(sym_freq) % 2 == 1 else ma_freq)[::-1]
        ])
        assert ma_log_amp.shape[0] == log_amp.shape[0], "`ma_log_amp` size does not match `log_amp` size."
        # compute residual spectrum and transform back to time domain
        res_amp = log_amp - ma_log_amp
        sr = np.abs(np.fft.ifft(np.exp(res_amp + 1j * phase)))
        return sr

    def compute_grads(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Slope of the straight line between different points of the time series
        multiplied by the average time step size.

        Parameters
        ----------
        X
            Time series of instances.
        t
            Time steps.

        Returns
        -------
        Array with slope values.
        """
        dX = X[-1] - X[-self.n_grad_points - 1:-1]
        dt = t[-1] - t[-self.n_grad_points - 1:-1]
        mean_grads = np.mean(dX / dt) * np.mean(dt)
        return mean_grads

    def add_est_points(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Pad the time series with additional points since the method works better if the anomaly point
        is towards the center of the sliding window.

        Parameters
        ----------
        X
            Time series of instances.
        t
            Time steps.

        Returns
        -------
        Padded version of X.
        """
        grads = self.compute_grads(X, t)
        X_add = X[-self.n_grad_points] + grads
        X_pad = np.concatenate([X, np.tile(X_add, self.n_est_points)])
        return X_pad

    def score(self, X: np.ndarray, t: np.ndarray = None) -> np.ndarray:
        """
        Compute outlier scores.

        Parameters
        ----------
        X
            Time series of instances.
        t
            Time steps.

        Returns
        -------
        Array with outlier scores for each instance in the batch.
        """
        if t is None:
            t = np.arange(X.shape[0])

        if len(X.shape) == 2:
            n_samples, n_dim = X.shape
            X = X.reshape(-1, )
            if X.shape[0] != n_samples:
                raise ValueError('Only univariate time series allowed for SR method. Number of features '
                                 'of time series equals {}.'.format(n_dim))

        X_pad = self.add_est_points(X, t)  # add padding
        sr = self.saliency_map(X_pad)      # compute saliency map
        sr = sr[:-self.n_est_points]       # remove padding again

        # pad the spectral residual before convolving. By applying `replicate` or `reflect` padding we can
        # remove some of the bias/outliers introduced at the beginning of the saliency map by a naive zero padding
        # performed by numpy. The reason for left padding is explained in a comment in the constructor.
        padded_sr = SpectralResidual.pad_same(X=sr,
                                              W=self.conv_local,
                                              method=self.padding_local_method,
                                              side=SIDE_LEFT)
        ma_sr = np.convolve(padded_sr, self.conv_local, 'valid')
        assert sr.shape[0] == ma_sr.shape[0], "`ma_sr` size does not match `sr` size."

        # compute the outlier score
        iscore = (sr - ma_sr) / (ma_sr + EPSILON)
        return iscore

    def predict(self,
                X: np.ndarray,
                t: np.ndarray = None,
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Compute outlier scores and transform into outlier predictions.

        Parameters
        ----------
        X
            Time series of instances.
        t
            Time steps.
        return_instance_score
            Whether to return instance level outlier scores.

        Returns
        -------
        Dictionary containing `meta` and `data` dictionaries.
         - `meta` - has the model's metadata.
         - `data` - contains the outlier predictions and instance level outlier scores.
        """
        if t is None:
            t = np.arange(X.shape[0])

        # compute outlier scores
        iscore = self.score(X.reshape(-1, ), t)

        # values above threshold are outliers
        outlier_pred = (iscore > self.threshold).astype(int)

        # populate output dict
        od = outlier_prediction_dict()
        od['meta'] = self.meta
        od['data']['is_outlier'] = outlier_pred
        if return_instance_score:
            od['data']['instance_score'] = iscore
        return od
