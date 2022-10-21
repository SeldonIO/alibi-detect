from copy import deepcopy
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.special import softmax
from typing import Callable, Dict, Optional, Union, Tuple
from alibi_detect.cd.base import BaseClassifierDrift
from alibi_detect.models.pytorch.trainer import trainer
from alibi_detect.utils.pytorch import get_device
from alibi_detect.utils.pytorch.data import TorchDataset
from alibi_detect.utils.pytorch.prediction import predict_batch
from alibi_detect.utils.warnings import deprecated_alias


class ClassifierDriftTorch(BaseClassifierDrift):
    @deprecated_alias(preprocess_x_ref='preprocess_at_init')
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            model: Union[nn.Module, nn.Sequential],
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
        Classifier-based drift detector. The classifier is trained on a fraction of the combined
        reference and test data and drift is detected on the remaining data. To use all the data
        to detect drift, a stratified cross-validation scheme can be chosen.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        model
            PyTorch classification model used for drift detection.
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
            Function to preprocess the data before computing the data drift metrics.
        preds_type
            Whether the model outputs 'probs' or 'logits'
        binarize_preds
            Whether to test for discrepency on soft (e.g. probs/logits) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
        reg_loss_fn
            The regularisation term reg_loss_fn(model) is added to the loss function being optimized.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the classifier.
            The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`.
        n_folds
            Optional number of stratified folds used for training. The model preds are then calculated
            on all the out-of-fold predictions. This allows to leverage all the reference and test data
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
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            x_ref_preprocessed=x_ref_preprocessed,
            preprocess_at_init=preprocess_at_init,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            preds_type=preds_type,
            binarize_preds=binarize_preds,
            train_size=train_size,
            n_folds=n_folds,
            retrain_from_scratch=retrain_from_scratch,
            seed=seed,
            input_shape=input_shape,
            data_type=data_type
        )

        if preds_type not in ['probs', 'logits']:
            raise ValueError("'preds_type' should be 'probs' or 'logits'")

        self.meta.update({'backend': 'pytorch'})

        # set device, define model and training kwargs
        self.device = get_device(device)
        self.original_model = model
        self.model = deepcopy(model)

        # define kwargs for dataloader and trainer
        self.loss_fn = nn.CrossEntropyLoss() if (self.preds_type == 'logits') else nn.NLLLoss()
        self.dataset = dataset
        self.dataloader = partial(dataloader, batch_size=batch_size, shuffle=True)
        self.predict_fn = partial(predict_batch, device=self.device,
                                  preprocess_fn=preprocess_batch_fn, batch_size=batch_size)
        self.train_kwargs = {'optimizer': optimizer, 'epochs': epochs,  'preprocess_fn': preprocess_batch_fn,
                             'reg_loss_fn': reg_loss_fn, 'learning_rate': learning_rate, 'verbose': verbose}
        if isinstance(train_kwargs, dict):
            self.train_kwargs.update(train_kwargs)

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Compute the out-of-fold drift metric such as the accuracy from a classifier
        trained to distinguish the reference data from the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value, a notion of distance between the trained classifier's out-of-fold performance \
        and that which we'd expect under the null assumption of no drift, \
        and the out-of-fold classifier model prediction probabilities on the reference and test data
        """
        x_ref, x = self.preprocess(x)
        x, y, splits = self.get_splits(x_ref, x)  # type: ignore

        # iterate over folds: train a new model for each fold and make out-of-fold (oof) predictions
        preds_oof_list, idx_oof_list = [], []
        for idx_tr, idx_te in splits:
            y_tr = y[idx_tr]
            if isinstance(x, np.ndarray):
                x_tr, x_te = x[idx_tr], x[idx_te]
            elif isinstance(x, list):
                x_tr, x_te = [x[_] for _ in idx_tr], [x[_] for _ in idx_te]
            else:
                raise TypeError(f'x needs to be of type np.ndarray or list and not {type(x)}.')
            ds_tr = self.dataset(x_tr, y_tr)
            dl_tr = self.dataloader(ds_tr)
            self.model = deepcopy(self.original_model) if self.retrain_from_scratch else self.model
            self.model = self.model.to(self.device)
            train_args = [self.model, self.loss_fn, dl_tr, self.device]
            trainer(*train_args, **self.train_kwargs)  # type: ignore
            preds = self.predict_fn(x_te, self.model.eval())
            preds_oof_list.append(preds)
            idx_oof_list.append(idx_te)
        preds_oof = np.concatenate(preds_oof_list, axis=0)
        probs_oof = softmax(preds_oof, axis=-1) if self.preds_type == 'logits' else preds_oof
        idx_oof = np.concatenate(idx_oof_list, axis=0)
        y_oof = y[idx_oof]
        n_cur = y_oof.sum()
        n_ref = len(y_oof) - n_cur
        p_val, dist = self.test_probs(y_oof, probs_oof, n_ref, n_cur)
        probs_sort = probs_oof[np.argsort(idx_oof)]
        return p_val, dist, probs_sort[:n_ref, 1], probs_sort[n_ref:, 1]
