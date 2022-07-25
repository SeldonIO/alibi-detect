import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow, BackendValidator
from alibi_detect.utils.warnings import deprecated_alias
from alibi_detect.base import DriftConfigMixin

if has_pytorch:
    from alibi_detect.cd.pytorch.lsdd import LSDDDriftTorch

if has_tensorflow:
    from alibi_detect.cd.tensorflow.lsdd import LSDDDriftTF


class LSDDDrift(DriftConfigMixin):
    @deprecated_alias(preprocess_x_ref='preprocess_at_init')
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            backend: str = 'tensorflow',
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            sigma: Optional[np.ndarray] = None,
            n_permutations: int = 100,
            n_kernel_centers: Optional[int] = None,
            lambda_rd_max: float = 0.2,
            device: Optional[str] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Least-squares density difference (LSDD) data drift detector using a permutation test.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        backend
            Backend used for the LSDD implementation.
        p_val
            p-value used for the significance of the permutation test.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        sigma
            Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple
            bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma`
            is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance
            between reference samples.
        n_permutations
            Number of permutations used in the permutation test.
        n_kernel_centers
            The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD.
            Defaults to 1/20th of the reference data.
        lambda_rd_max
            The maximum relative difference between two estimates of LSDD that the regularization parameter
            lambda is allowed to cause. Defaults to 0.2 as in the paper.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
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
            backend_options={'tensorflow': ['tensorflow'],
                             'pytorch': ['pytorch']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        kwargs = locals()
        args = [kwargs['x_ref']]
        pop_kwargs = ['self', 'x_ref', 'backend', '__class__']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if backend == 'tensorflow' and has_tensorflow:
            kwargs.pop('device', None)
            self._detector = LSDDDriftTF(*args, **kwargs)  # type: ignore
        else:
            self._detector = LSDDDriftTorch(*args, **kwargs)  # type: ignore
        self.meta = self._detector.meta

    def predict(self, x: Union[np.ndarray, list], return_p_val: bool = True, return_distance: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_p_val
            Whether to return the p-value of the permutation test.
        return_distance
            Whether to return the LSDD metric between the new batch and reference data.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold and LSDD metric.
        """
        return self._detector.predict(x, return_p_val, return_distance)

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, float]:
        """
        Compute the p-value resulting from a permutation test using the least-squares density
        difference as a distance measure between the reference data and the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the LSDD between the reference and test set,
        and the LSDD threshold above which drift is flagged.
        """
        return self._detector.score(x)

    def get_config(self) -> dict:  # Needed due to need to unnormalize x_ref
        """
        Get the detector's configuration dictionary.

        Returns
        -------
        The detector's configuration dictionary.
        """
        cfg = super().get_config()
        # Unnormalize x_ref
        if self._detector.preprocess_at_init or self._detector.preprocess_fn is None \
                or self._detector.x_ref_preprocessed:
            cfg['x_ref'] = self._detector._unnormalize(cfg['x_ref'])
        return cfg
