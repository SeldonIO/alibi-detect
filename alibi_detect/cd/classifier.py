from functools import partial
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from typing import Callable, Dict, Optional, Tuple, Union
from alibi_detect.base import BaseDetector, concept_drift_dict
from alibi_detect.cd.utils import update_reference
from alibi_detect.utils.metrics import accuracy

logger = logging.getLogger(__name__)


class ClassifierDrift(BaseDetector):

    def __init__(self,
                 threshold: float = None,
                 model: Union[tf.keras.Model, tf.keras.Sequential] = None,
                 X_ref: Union[np.ndarray, list] = None,
                 preprocess_X_ref: bool = True,
                 update_X_ref: Optional[Dict[str, int]] = None,
                 preprocess_fn: Optional[Callable] = None,
                 preprocess_kwargs: Optional[dict] = None,
                 metric_fn: Callable = accuracy,
                 metric_name: Optional[str] = None,
                 train_size: Optional[float] = .75,
                 n_folds: Optional[int] = None,
                 seed: int = 0,
                 optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
                 compile_kwargs: Optional[dict] = None,
                 batch_size: int = 32,
                 epochs: int = 3,
                 verbose: int = 0,
                 fit_kwargs: Optional[dict] = None,
                 data_type: Optional[str] = None
                 ) -> None:
        """
        Classifier-based drift detector. The classifier is trained on a fraction of the combined
        reference and test data and drift is detected on the remaining data. To use all the data
        to detect drift, a stratified cross-validation scheme can be chosen. The metric to evaluate
        the drift detector defaults to accuracy but is flexible and can be changed to any function
        to handle e.g. imbalanced splits between reference and test sets.

        Parameters
        ----------
        threshold
            Threshold for the drift metric (default is accuracy). Values above the threshold are
            classified as drift.
        model
            Classification model used for drift detection.
        X_ref
            Data used as reference distribution. Can be a list for text data which is then turned into an array
            after the preprocessing step.
        preprocess_X_ref
            Whether to already preprocess and store the reference data.
        update_X_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        preprocess_kwargs
            Kwargs for `preprocess_fn`.
        metric_fn
            Function computing the drift metric. Takes `y_true` and `y_pred` as input and
            returns a float: metric_fn(y_true, y_pred). Defaults to accuracy.
        metric_name
            Optional name for the metric_fn used in the return dict. Defaults to 'metric_fn.__name__'.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the classifier.
            The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`.
        n_folds
            Optional number of stratified folds used for training. The metric is then calculated
            on all the out-of-fold predictions. This allows to leverage all the reference and test data
            for drift detection at the expense of longer computation. If both `train_size` and `n_folds`
            are specified, `n_folds` is prioritized.
        seed
            Optional random seed for fold selection.
        optimizer
            Optimizer used during training of the classifier.
        compile_kwargs
            Optional additional kwargs when compiling the classifier.
        batch_size
            Batch size used during training of the classifier.
        epochs
            Number of training epochs for the classifier for each (optional) fold.
        verbose
            Verbosity level during the training of the classifier.
            0 is silent, 1 a progress bar and 2 prints the statistics after each epoch.
        fit_kwargs
            Optional additional kwargs when fitting the classifier.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('Need to set drift threshold to detect data drift.')

        if isinstance(train_size, float) and isinstance(n_folds, int):
            logger.warning('Both `n_folds` and `train_size` specified. By default `n_folds` is used.')

        if isinstance(preprocess_fn, Callable) and isinstance(preprocess_kwargs, dict):  # type: ignore
            self.preprocess_fn = partial(preprocess_fn, **preprocess_kwargs)
        else:
            self.preprocess_fn = preprocess_fn  # type: ignore

        # optionally already preprocess reference data
        self.preprocess_X_ref = preprocess_X_ref
        if preprocess_X_ref and isinstance(self.preprocess_fn, Callable):  # type: ignore
            self.X_ref = self.preprocess_fn(X_ref)
        else:
            self.X_ref = X_ref
        self.update_X_ref = update_X_ref
        self.n = X_ref.shape[0]  # type: ignore

        self.threshold = threshold
        self.metric_fn = metric_fn
        if isinstance(n_folds, int):
            self.train_size = None
            self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            self.train_size, self.skf = train_size, None

        self.model = model
        self.compile_kwargs = {'optimizer': optimizer, 'loss': BinaryCrossentropy()}
        if isinstance(compile_kwargs, dict):
            self.compile_kwargs.update(compile_kwargs)
        self.fit_kwargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': verbose}
        if isinstance(fit_kwargs, dict):
            self.fit_kwargs.update(fit_kwargs)

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type
        self.metric_name = metric_fn.__name__ if metric_name is None else metric_name
        self.meta['params'] = {'metric_fn': self.metric_name}

    def preprocess(self, X: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data preprocessing before computing the drift scores.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Preprocessed reference data and new instances.
        """
        if isinstance(self.preprocess_fn, Callable):  # type: ignore
            X = self.preprocess_fn(X)
            X_ref = self.X_ref if self.preprocess_X_ref else self.preprocess_fn(self.X_ref)
            return X_ref, X
        else:
            return self.X_ref, X

    def score(self, X: Union[np.ndarray, list]) -> float:
        """
        Compute the out-of-fold drift metric such as the accuracy from a classifier
        trained to distinguish the reference data from the data to be tested.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Drift metric (e.g. accuracy) obtained from out-of-fold predictions from a trained classifier.
        """
        X_ref, X = self.preprocess(X)

        # create dataset and labels
        x = np.concatenate([X_ref, X], axis=0)
        y = np.concatenate([np.zeros(X_ref.shape[0]), np.ones(X.shape[0])], axis=0).astype(int)

        # random shuffle if stratified folds are not used
        if self.skf is None:
            n_tot = x.shape[0]
            idx_shuffle = np.random.choice(np.arange(x.shape[0]), size=n_tot, replace=False)
            n_tr = int(n_tot * self.train_size)
            idx_tr, idx_te = idx_shuffle[:n_tr], idx_shuffle[n_tr:]
            splits = [(idx_tr, idx_te)]
        else:  # use stratified folds
            splits = self.skf.split(x, y)

        # iterate over folds: train a new model for each fold and make out-of-fold (oof) predictions
        preds_oof, idx_oof = [], []
        for idx_tr, idx_te in splits:
            x_tr, y_tr, x_te = x[idx_tr], np.eye(2)[y[idx_tr]], x[idx_te]
            clf = tf.keras.models.clone_model(self.model)
            clf.compile(**self.compile_kwargs)
            clf.fit(x=x_tr, y=y_tr, **self.fit_kwargs)
            preds = clf.predict(x_te, batch_size=self.fit_kwargs['batch_size'])
            preds_oof.append(preds)
            idx_oof.append(idx_te)
        preds_oof = np.concatenate(preds_oof, axis=0)[:, 1]
        idx_oof = np.concatenate(idx_oof, axis=0)
        drift_metric = self.metric_fn(y[idx_oof], preds_oof)
        return drift_metric

    def predict(self, X: Union[np.ndarray, list], return_metric: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        X
            Batch of instances.
        return_metric
            Whether to return the drift metric from the detector.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the drift metric and threshold.
        """
        # compute drift scores
        drift_metric = self.score(X)
        drift_pred = int(drift_metric > self.threshold)

        # update reference dataset
        if isinstance(self.update_X_ref, dict) and self.preprocess_fn is not None and self.preprocess_X_ref:
            X = self.preprocess_fn(X)
        self.X_ref = update_reference(self.X_ref, X, self.n, self.update_X_ref)
        # used for reservoir sampling
        self.n += X.shape[0]  # type: ignore

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_metric:
            cd['data'][self.metric_name] = drift_metric
            cd['data']['threshold'] = self.threshold
        return cd
