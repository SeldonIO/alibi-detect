import numpy as np
from typing import Callable, Dict, Optional, Union
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

if has_pytorch:
    from torch.utils.data import DataLoader
    from alibi_detect.cd.pytorch.classifier import ClassifierDriftTorch
    from alibi_detect.utils.pytorch.data import TorchDataset

if has_tensorflow:
    from alibi_detect.cd.tensorflow.classifier import ClassifierDriftTF
    from alibi_detect.utils.tensorflow.data import TFDataset


class ClassifierDrift:
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            model: Callable,
            backend: str = 'tensorflow',
            p_val: float = .05,
            preprocess_x_ref: bool = True,
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
            PyTorch or TensorFlow classification model used for drift detection.
        backend
            Backend used for the training loop implementation.
        p_val
            p-value used for the significance of the test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        preds_type
            Whether the model outputs 'probs' or 'logits'
        binarize_preds
            Whether to test for discrepency on soft  (e.g. probs/logits) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
        reg_loss_fn
            The regularisation term reg_loss_fn(model) is added to the loss function being optimized.
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
            Optimizer used during training of the classifier.
        learning_rate
            Learning rate used by optimizer.
        batch_size
            Batch size used during training of the classifier.
        preprocess_batch_fn
            Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
            processed by the model.
        epochs
            Number of training epochs for the classifier for each (optional) fold.
        verbose
            Verbosity level during the training of the classifier. 0 is silent, 1 a progress bar.
        train_kwargs
            Optional additional kwargs when fitting the classifier.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
        dataset
            Dataset object used during training.
        dataloader
            Dataloader object used during training. Only relevant for 'pytorch' backend.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        backend = backend.lower()
        if backend == 'tensorflow' and not has_tensorflow or backend == 'pytorch' and not has_pytorch:
            raise ImportError(f'{backend} not installed. Cannot initialize and run the '
                              f'ClassifierDrift detector with {backend} backend.')
        elif backend not in ['tensorflow', 'pytorch']:
            raise NotImplementedError(f'{backend} not implemented. Use tensorflow or pytorch instead.')

        kwargs = locals()
        args = [kwargs['x_ref'], kwargs['model']]
        pop_kwargs = ['self', 'x_ref', 'model', 'backend', '__class__']
        if kwargs['optimizer'] is None:
            pop_kwargs += ['optimizer']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if backend == 'tensorflow' and has_tensorflow:
            pop_kwargs = ['device', 'dataloader']
            [kwargs.pop(k, None) for k in pop_kwargs]
            if dataset is None:
                kwargs.update({'dataset': TFDataset})
            self._detector = ClassifierDriftTF(*args, **kwargs)  # type: ignore
        else:
            if dataset is None:
                kwargs.update({'dataset': TorchDataset})
            if dataloader is None:
                kwargs.update({'dataloader': DataLoader})
            self._detector = ClassifierDriftTorch(*args, **kwargs)  # type: ignore
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
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, performance of the classifier
        relative to its expectation under the no-change null, the out-of-fold classifier model
        prediction probabilities on the reference and test data, and the trained model.
        """
        return self._detector.predict(x, return_p_val, return_distance, return_probs, return_model)
