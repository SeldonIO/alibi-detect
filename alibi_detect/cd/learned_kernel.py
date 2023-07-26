import numpy as np
from typing import Callable, Dict, Optional, Union
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow, has_keops, BackendValidator, Framework
from alibi_detect.utils.warnings import deprecated_alias
from alibi_detect.base import DriftConfigMixin
from alibi_detect.utils._types import TorchDeviceType

if has_pytorch:
    from torch.utils.data import DataLoader
    from alibi_detect.cd.pytorch.learned_kernel import LearnedKernelDriftTorch
    from alibi_detect.utils.pytorch.data import TorchDataset

if has_tensorflow:
    from alibi_detect.cd.tensorflow.learned_kernel import LearnedKernelDriftTF
    from alibi_detect.utils.tensorflow.data import TFDataset

if has_keops:
    from alibi_detect.cd.keops.learned_kernel import LearnedKernelDriftKeops


class LearnedKernelDrift(DriftConfigMixin):
    @deprecated_alias(preprocess_x_ref='preprocess_at_init')
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            kernel: Callable,
            backend: str = 'tensorflow',
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            n_permutations: int = 100,
            batch_size_permutations: int = 1000000,
            var_reg: float = 1e-5,
            reg_loss_fn: Callable = (lambda kernel: 0),
            train_size: Optional[float] = .75,
            retrain_from_scratch: bool = True,
            optimizer: Optional[Callable] = None,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            batch_size_predict: int = 32,
            preprocess_batch_fn: Optional[Callable] = None,
            epochs: int = 3,
            num_workers: int = 0,
            verbose: int = 0,
            train_kwargs: Optional[dict] = None,
            device: TorchDeviceType = None,
            dataset: Optional[Callable] = None,
            dataloader: Optional[Callable] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) data drift detector where the kernel is trained to maximise an
        estimate of the test power. The kernel is trained on a split of the reference and test instances
        and then the MMD is evaluated on held out instances and a permutation test is performed.

        For details see Liu et al (2020): Learning Deep Kernels for Non-Parametric Two-Sample Tests
        (https://arxiv.org/abs/2002.09116)

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        kernel
            Trainable PyTorch or TensorFlow module that returns a similarity between two instances.
        backend
            Backend used by the kernel and training loop.
        p_val
            p-value used for the significance of the test.
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
            Function to preprocess the data before applying the kernel.
        n_permutations
            The number of permutations to use in the permutation test once the MMD has been computed.
        batch_size_permutations
            KeOps computes the n_permutations of the MMD^2 statistics in chunks of batch_size_permutations.
            Only relevant for 'keops' backend.
        var_reg
            Constant added to the estimated variance of the MMD for stability.
        reg_loss_fn
            The regularisation term reg_loss_fn(kernel) is added to the loss function being optimized.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the kernel.
            The drift is detected on `1 - train_size`.
        retrain_from_scratch
            Whether the kernel should be retrained from scratch for each set of test data or whether
            it should instead continue training from where it left off on the previous set.
        optimizer
            Optimizer used during training of the kernel.
        learning_rate
            Learning rate used by optimizer.
        batch_size
            Batch size used during training of the kernel.
        batch_size_predict
            Batch size used for the trained drift detector predictions.
        preprocess_batch_fn
            Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
            processed by the kernel.
        epochs
            Number of training epochs for the kernel. Corresponds to the smaller of the reference and test sets.
        num_workers
            Number of workers for the dataloader. The default (`num_workers=0`) means multi-process data loading
            is disabled. Setting `num_workers>0` may be unreliable on Windows.
        verbose
            Verbosity level during the training of the kernel. 0 is silent, 1 a progress bar.
        train_kwargs
            Optional additional kwargs when training the kernel.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``. Relevant for 'pytorch' and 'keops' backends.
        dataset
            Dataset object used during training.
        dataloader
            Dataloader object used during training. Relevant for 'pytorch' and 'keops' backends.
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
                             Framework.PYTORCH: [Framework.PYTORCH],
                             Framework.KEOPS: [Framework.KEOPS]},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        kwargs = locals()
        args = [kwargs['x_ref'], kwargs['kernel']]
        pop_kwargs = ['self', 'x_ref', 'kernel', 'backend', '__class__']
        if kwargs['optimizer'] is None:
            pop_kwargs += ['optimizer']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if backend == Framework.TENSORFLOW:
            pop_kwargs = ['device', 'dataloader', 'batch_size_permutations', 'num_workers']
            [kwargs.pop(k, None) for k in pop_kwargs]
            if dataset is None:
                kwargs.update({'dataset': TFDataset})
            detector = LearnedKernelDriftTF
        else:
            if dataset is None:
                kwargs.update({'dataset': TorchDataset})
            if dataloader is None:
                kwargs.update({'dataloader': DataLoader})
            if backend == Framework.PYTORCH:
                pop_kwargs = ['batch_size_permutations']
                [kwargs.pop(k, None) for k in pop_kwargs]
                detector = LearnedKernelDriftTorch
            else:
                detector = LearnedKernelDriftKeops

        self._detector = detector(*args, **kwargs)
        self.meta = self._detector.meta

    def predict(self, x: Union[np.ndarray, list],  return_p_val: bool = True,
                return_distance: bool = True, return_kernel: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float, Callable]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_p_val
            Whether to return the p-value of the permutation test.
        return_distance
            Whether to return the MMD metric between the new batch and reference data.
        return_kernel
            Whether to return the updated kernel trained to discriminate reference and test instances.

        Returns
        -------
        Dictionary containing ``'meta'`` and ``'data'`` dictionaries.
            - ``'meta'`` has the detector's metadata.
            - ``'data'`` contains the drift prediction and optionally the p-value, threshold, MMD metric and \
            trained kernel.
        """
        return self._detector.predict(x, return_p_val, return_distance, return_kernel)
