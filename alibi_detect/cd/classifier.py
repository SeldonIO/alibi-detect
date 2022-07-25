import numpy as np
from typing import Callable, Dict, Optional, Union
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow, \
    BackendValidator
from alibi_detect.base import DriftConfigMixin


from sklearn.base import ClassifierMixin
from alibi_detect.cd.sklearn.classifier import ClassifierDriftSklearn

if has_pytorch:
    from torch.utils.data import DataLoader
    from alibi_detect.cd.pytorch.classifier import ClassifierDriftTorch
    from alibi_detect.utils.pytorch.data import TorchDataset

if has_tensorflow:
    from alibi_detect.cd.tensorflow.classifier import ClassifierDriftTF
    from alibi_detect.utils.tensorflow.data import TFDataset


class ClassifierDrift(DriftConfigMixin):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            model: Union[ClassifierMixin, Callable],
            backend: str = 'tensorflow',
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            preds_type: str = 'probs',
            binarize_preds: bool = False,
            reg_loss_fn: Callable = (lambda model: 0),
            train_size: Optional[float] = .75,
            n_folds: Optional[int] = None,
            retrain_from_scratch: bool = True,
            seed: int = 0,
            optimizer: Optional[Callable] = None,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            preprocess_batch_fn: Optional[Callable] = None,
            epochs: int = 3,
            verbose: int = 0,
            train_kwargs: Optional[dict] = None,
            device: Optional[str] = None,
            dataset: Optional[Callable] = None,
            dataloader: Optional[Callable] = None,
            input_shape: Optional[tuple] = None,
            use_calibration: bool = False,
            calibration_kwargs: Optional[dict] = None,
            use_oob: bool = False,
            data_type: Optional[str] = None
    ) -> None:
        """
        Classifier-based drift detector. The classifier is trained on a fraction of the combined
        reference and test data and drift is detected on the remaining data. To use all the data
        to detect drift, a stratified cross-validation scheme can be chosen.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        model
            PyTorch, TensorFlow or Sklearn classification model used for drift detection.
        backend
            Backend used for the training loop implementation. Supported: 'tensorflow' | 'pytorch' | 'sklearn'.
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
            Reference data can optionally be updated to the last `n` instances seen by the detector
            or via reservoir sampling with size `n`. For the former, the parameter equals `{'last': n}` while
            for reservoir sampling `{'reservoir_sampling': n}` is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        preds_type
            Whether the model outputs 'probs' (probabilities - for 'tensorflow', 'pytorch', 'sklearn' models),
            'logits' (for 'pytorch', 'tensorflow' models), 'scores' (for 'sklearn' models if `decision_function`
            is supported).
        binarize_preds
            Whether to test for discrepancy on soft  (e.g. probs/logits/scores) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
        reg_loss_fn
            The regularisation term `reg_loss_fn(model)` is added to the loss function being optimized.
            Only relevant for 'tensorflow` and 'pytorch' backends.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the classifier.
            The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`.
        n_folds
            Optional number of stratified folds used for training. The model preds are then calculated
            on all the out-of-fold instances. This allows to leverage all the reference and test data
            for drift detection at the expense of longer computation. If both `train_size` and `n_folds`
            are specified, `n_folds` is prioritized.
        retrain_from_scratch
            Whether the classifier should be retrained from scratch for each set of test data or whether
            it should instead continue training from where it left off on the previous set.
        seed
            Optional random seed for fold selection.
        optimizer
            Optimizer used during training of the classifier. Only relevant for 'tensorflow' and 'pytorch' backends.
        learning_rate
            Learning rate used by optimizer. Only relevant for 'tensorflow' and 'pytorch' backends.
        batch_size
            Batch size used during training of the classifier. Only relevant for 'tensorflow' and 'pytorch' backends.
        preprocess_batch_fn
            Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
            processed by the model. Only relevant for 'tensorflow' and 'pytorch' backends.
        epochs
            Number of training epochs for the classifier for each (optional) fold. Only relevant for 'tensorflow'
            and 'pytorch' backends.
        verbose
            Verbosity level during the training of the classifier. 0 is silent, 1 a progress bar. Only relevant for
            'tensorflow' and 'pytorch' backends.
        train_kwargs
            Optional additional kwargs when fitting the classifier. Only relevant for 'tensorflow' and
            'pytorch' backends.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
        dataset
            Dataset object used during training. Only relevant for 'tensorflow' and 'pytorch' backends.
        dataloader
            Dataloader object used during training. Only relevant for 'pytorch' backend.
        input_shape
            Shape of input data.
        use_calibration
            Whether to use calibration. Calibration can be used on top of any model.
            Only relevant for 'sklearn' backend.
        calibration_kwargs
            Optional additional kwargs for calibration. Only relevant for 'sklearn' backend.
            See https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
            for more details.
        use_oob
            Whether to use out-of-bag(OOB) predictions. Supported only for `RandomForestClassifier`.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        # Set config
        self._set_config(locals())

        backend = backend.lower()
        BackendValidator(
            backend_options={'tensorflow': ['tensorflow'],
                             'pytorch': ['pytorch'],
                             'sklearn': ['sklearn']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        kwargs = locals()
        args = [kwargs['x_ref'], kwargs['model']]
        pop_kwargs = ['self', 'x_ref', 'model', 'backend', '__class__']
        if kwargs['optimizer'] is None:
            pop_kwargs += ['optimizer']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if backend == 'tensorflow':
            pop_kwargs = ['device', 'dataloader', 'use_calibration', 'calibration_kwargs', 'use_oob']
            [kwargs.pop(k, None) for k in pop_kwargs]
            if dataset is None:
                kwargs.update({'dataset': TFDataset})
            self._detector = ClassifierDriftTF(*args, **kwargs)  # type: ignore
        elif backend == 'pytorch':
            pop_kwargs = ['use_calibration', 'calibration_kwargs', 'use_oob']
            [kwargs.pop(k, None) for k in pop_kwargs]
            if dataset is None:
                kwargs.update({'dataset': TorchDataset})
            if dataloader is None:
                kwargs.update({'dataloader': DataLoader})
            self._detector = ClassifierDriftTorch(*args, **kwargs)  # type: ignore
        else:
            pop_kwargs = ['reg_loss_fn', 'optimizer', 'learning_rate', 'batch_size', 'preprocess_batch_fn',
                          'epochs', 'train_kwargs', 'device',  'dataset', 'dataloader', 'verbose']
            [kwargs.pop(k, None) for k in pop_kwargs]
            self._detector = ClassifierDriftSklearn(*args, **kwargs)  # type: ignore
        self.meta = self._detector.meta

    def predict(self, x: Union[np.ndarray, list],  return_p_val: bool = True,
                return_distance: bool = True, return_probs: bool = True, return_model: bool = True) \
            -> Dict[str, Dict[str, Union[str, int, float, Callable]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_p_val
            Whether to return the p-value of the test.
        return_distance
            Whether to return a notion of strength of the drift.
            K-S test stat if binarize_preds=False, otherwise relative error reduction.
        return_probs
            Whether to return the instance level classifier probabilities for the reference and test data
            (0=reference data, 1=test data).
        return_model
            Whether to return the updated model trained to discriminate reference and test instances.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries

         - 'meta' - has the model's metadata.

         - 'data' - contains the drift prediction and optionally the p-value, performance of the classifier \
        relative to its expectation under the no-change null, the out-of-fold classifier model \
        prediction probabilities on the reference and test data, and the trained model. \
        """
        return self._detector.predict(x, return_p_val, return_distance, return_probs, return_model)
