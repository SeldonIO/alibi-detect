import os
import numpy as np
from typing import Any, Callable, Dict, Optional, Union
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow, BackendValidator, Framework
from alibi_detect.base import DriftConfigMixin
from alibi_detect.utils._types import TorchDeviceType

if has_pytorch:
    from alibi_detect.cd.pytorch.mmd_online import MMDDriftOnlineTorch

if has_tensorflow:
    from alibi_detect.cd.tensorflow.mmd_online import MMDDriftOnlineTF


class MMDDriftOnline(DriftConfigMixin):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: int,
            backend: str = 'tensorflow',
            preprocess_fn: Optional[Callable] = None,
            x_ref_preprocessed: bool = False,
            kernel: Optional[Callable] = None,
            sigma: Optional[np.ndarray] = None,
            n_bootstraps: int = 1000,
            device: TorchDeviceType = None,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Online maximum Mean Discrepancy (MMD) data drift detector using preconfigured thresholds.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined
            as the expected run-time from t=0.
        window_size
            The size of the sliding test-window used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        backend
            Backend used for the MMD implementation and configuration.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        kernel
            Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
        sigma
            Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array.
            The kernel evaluation is then averaged over those bandwidths. If `sigma` is not specified, the 'median
            heuristic' is adopted whereby `sigma` is set as the median pairwise distance between reference samples.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ERT.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``. Only relevant for 'pytorch' backend.
        verbose
            Whether or not to print progress during configuration.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        # Set config
        self._set_config(locals())

        backend = backend.lower()
        BackendValidator(
            backend_options={Framework.TENSORFLOW: [Framework.TENSORFLOW],
                             Framework.PYTORCH: [Framework.PYTORCH]},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        kwargs = locals()
        args = [kwargs['x_ref'], kwargs['ert'], kwargs['window_size']]
        pop_kwargs = ['self', 'x_ref', 'ert', 'window_size', 'backend', '__class__']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if kernel is None:
            if backend == Framework.TENSORFLOW:
                from alibi_detect.utils.tensorflow.kernels import GaussianRBF
            else:
                from alibi_detect.utils.pytorch.kernels import GaussianRBF  # type: ignore
            kwargs.update({'kernel': GaussianRBF})

        if backend == Framework.TENSORFLOW:
            kwargs.pop('device', None)
            self._detector = MMDDriftOnlineTF(*args, **kwargs)
        else:
            self._detector = MMDDriftOnlineTorch(*args, **kwargs)  # type: ignore
        self.meta = self._detector.meta

    @property
    def t(self):
        return self._detector.t

    @property
    def test_stats(self):
        return self._detector.test_stats

    @property
    def thresholds(self):
        return [self._detector.thresholds[min(s, self._detector.window_size-1)] for s in range(self.t)]

    def reset_state(self):
        """
        Resets the detector to its initial state (`t=0`). This does not include reconfiguring thresholds.
        """
        self._detector.reset_state()

    def predict(self, x_t: Union[np.ndarray, Any], return_test_stat: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether the most recent window of data has drifted from the reference data.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.
        return_test_stat
            Whether to return the test statistic (squared MMD) and threshold.

        Returns
        -------
        Dictionary containing ``'meta'`` and ``'data'`` dictionaries.
            - ``'meta'`` has the model's metadata.
            - ``'data'`` contains the drift prediction and optionally the test-statistic and threshold.
        """
        return self._detector.predict(x_t, return_test_stat)

    def score(self, x_t: Union[np.ndarray, Any]) -> float:
        """
        Compute the test-statistic (squared MMD) between the reference window and test window.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.

        Returns
        -------
        Squared MMD estimate between reference window and test window.
        """
        return self._detector.score(x_t)

    def save_state(self, filepath: Union[str, os.PathLike]):
        """
        Save a detector's state to disk in order to generate a checkpoint.

        Parameters
        ----------
        filepath
            The directory to save state to.
        """
        self._detector.save_state(filepath)

    def load_state(self, filepath: Union[str, os.PathLike]):
        """
        Load the detector's state from disk, in order to restart from a checkpoint previously generated with
        `save_state`.

        Parameters
        ----------
        filepath
            The directory to load state from.
        """
        self._detector.load_state(filepath)

    def get_config(self) -> dict:  # Needed due to self.x_ref being a torch.Tensor when backend='pytorch'
        """
        Get the detector's configuration dictionary.

        Returns
        -------
        The detector's configuration dictionary.
        """
        cfg = super().get_config()
        if cfg.get('backend') == 'pytorch':
            cfg['x_ref'] = cfg['x_ref'].detach().cpu().numpy()
        return cfg
