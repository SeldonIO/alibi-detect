import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Dict, Optional, Union
from alibi_detect.cd.pytorch.classifier import ClassifierDriftTorch
from alibi_detect.utils.pytorch.data import TorchDataset
from alibi_detect.utils.pytorch import GaussianRBF
from alibi_detect.utils.pytorch.prediction import predict_batch

logger = logging.getLogger(__name__)


class SpotTheDiffDriftTorch:
    def __init__(
            self,
            x_ref: np.ndarray,
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_fn: Optional[Callable] = None,
            kernel: Optional[nn.Module] = None,
            n_diffs: int = 1,
            initial_diffs: Optional[np.ndarray] = None,
            l1_reg: float = 0.01,
            binarize_preds: bool = False,
            train_size: Optional[float] = .75,
            n_folds: Optional[int] = None,
            retrain_from_scratch: bool = True,
            seed: int = 0,
            optimizer: Callable = torch.optim.Adam,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            preprocess_batch_fn: Optional[Callable] = None,
            epochs: int = 3,
            verbose: int = 0,
            train_kwargs: Optional[dict] = None,
            device: Optional[str] = None,
            dataset: Callable = TorchDataset,
            dataloader: Callable = DataLoader,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Classifier-based drift detector with a classifier of form y = a + b_1*k(x,w_1) + ... + b_J*k(x,w_J),
        where k is a kernel and w_1,...,w_J are learnable test locations. If drift has occured the test locations
        learn to be more/less (given by sign of b_i) similar to test instances than reference instances.
        The test locations are regularised to be close to the average reference instance such that the **difference**
        is then interpretable as the transformation required for each feature to make the average instance more/less
        like a test instance than a reference instance.

        The classifier is trained on a fraction of the combined reference and test data and drift is detected on
        the remaining data. To use all the data to detect drift, a stratified cross-validation scheme can be chosen.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for the significance of the test.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        kernel
            Differentiable Pytorch model used to define similarity between instances, defaults to Gaussian RBF.
        n_diffs
            The number of test locations to use, each corresponding to an interpretable difference.
        initial_diffs
            Array used to initialise the diffs that will be learned. Defaults to Gaussian
            for each feature with equal variance to that of reference data.
        l1_reg
            Strength of l1 regularisation to apply to the differences.
        binarize_preds
            Whether to test for discrepency on soft (e.g. probs/logits) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
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
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'.
        dataset
            Dataset object used during training.
        dataloader
            Dataloader object used during training.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        if preprocess_fn is not None and preprocess_batch_fn is not None:
            raise ValueError("SpotTheDiffDrift detector only supports preprocess_fn or preprocess_batch_fn, not both.")
        if n_folds is not None and n_folds > 1:
            logger.warning("When using multiple folds the returned diffs will correspond to the final fold only.")

        if not x_ref_preprocessed and preprocess_fn is not None:
            x_ref_proc = preprocess_fn(x_ref)
        elif not x_ref_preprocessed and preprocess_batch_fn is not None:
            x_ref_proc = predict_batch(
                x_ref, lambda x: x, preprocess_fn=preprocess_batch_fn,
                device=torch.device('cpu'), batch_size=batch_size
            )
        else:
            x_ref_proc = x_ref

        if kernel is None:
            kernel = GaussianRBF(trainable=True)
        if initial_diffs is None:
            initial_diffs = np.random.normal(size=(n_diffs,) + x_ref_proc.shape[1:]) * x_ref_proc.std(0)
        else:
            if len(initial_diffs) != n_diffs:
                raise ValueError("Should have initial_diffs.shape[0] == n_diffs")

        model = SpotTheDiffDriftTorch.InterpretableClf(kernel, x_ref_proc, initial_diffs)
        reg_loss_fn = (lambda model: model.diffs.abs().mean() * l1_reg)

        self._detector = ClassifierDriftTorch(
            x_ref=x_ref,
            model=model,
            p_val=p_val,
            x_ref_preprocessed=x_ref_preprocessed,
            preprocess_at_init=True,
            update_x_ref=None,
            preprocess_fn=preprocess_fn,
            preds_type='logits',
            binarize_preds=binarize_preds,
            reg_loss_fn=reg_loss_fn,
            train_size=train_size,
            n_folds=n_folds,
            retrain_from_scratch=retrain_from_scratch,
            seed=seed,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            preprocess_batch_fn=preprocess_batch_fn,
            epochs=epochs,
            verbose=verbose,
            train_kwargs=train_kwargs,
            device=device,
            dataset=dataset,
            dataloader=dataloader,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta = self._detector.meta
        self.meta['params']['name'] = 'SpotTheDiffDrift'
        self.meta['params']['n_diffs'] = n_diffs
        self.meta['params']['l1_reg'] = l1_reg
        self.meta['params']['initial_diffs'] = initial_diffs

    class InterpretableClf(nn.Module):
        def __init__(self, kernel: nn.Module, x_ref: np.ndarray, initial_diffs: np.ndarray):
            super().__init__()
            self.kernel = kernel
            self.mean = nn.Parameter(torch.as_tensor(x_ref.mean(0)), requires_grad=False)
            self.diffs = nn.Parameter(torch.as_tensor(initial_diffs, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros((1,)))
            self.coeffs = nn.Parameter(torch.zeros((len(initial_diffs),)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            k_xtl = self.kernel(x, self.mean + self.diffs)
            logits = self.bias + k_xtl @ self.coeffs[:, None]
            return torch.cat([-logits, logits], 1)

    def predict(
        self, x: np.ndarray,  return_p_val: bool = True, return_distance: bool = True,
        return_probs: bool = True, return_model: bool = False
    ) -> Dict[str, Dict[str, Union[str, int, float, Callable]]]:
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
        'meta' has the detector's metadata.
        'data' contains the drift prediction, the diffs used to distinguish reference from test instances,
        and optionally the p-value, performance of the classifier relative to its expectation under the
        no-change null, the out-of-fold classifier model prediction probabilities on the reference and test
        data, and the trained model.
        """
        preds = self._detector.predict(x, return_p_val, return_distance, return_probs, return_model=True)
        preds['data']['diffs'] = preds['data']['model'].diffs.detach().cpu().numpy()  # type: ignore
        preds['data']['diff_coeffs'] = preds['data']['model'].coeffs.detach().cpu().numpy()  # type: ignore
        if not return_model:
            del preds['data']['model']
        return preds
