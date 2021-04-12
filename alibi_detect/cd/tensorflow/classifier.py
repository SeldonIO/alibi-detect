import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from typing import Callable, Dict, Optional, Union
from alibi_detect.cd.base import BaseClassifierDrift
from alibi_detect.utils.metrics import accuracy


class ClassifierDriftTF(BaseClassifierDrift):
    def __init__(
            self,
            x_ref: np.ndarray,
            model: Union[tf.keras.Model, tf.keras.Sequential],
            threshold: float = .55,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            metric_fn: Callable = accuracy,
            metric_name: Optional[str] = None,
            train_size: Optional[float] = .75,
            n_folds: Optional[int] = None,
            seed: int = 0,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam,
            learning_rate: float = 1e-3,
            compile_kwargs: Optional[dict] = None,
            batch_size: int = 32,
            epochs: int = 3,
            verbose: int = 0,
            train_kwargs: Optional[dict] = None,
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
            TensorFlow classification model used for drift detection.
        threshold
            Threshold for the drift metric (default is accuracy). Values above the threshold are
            classified as drift.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
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
        learning_rate
            Learning rate used by optimizer.
        compile_kwargs
            Optional additional kwargs when compiling the classifier.
        batch_size
            Batch size used during training of the classifier.
        epochs
            Number of training epochs for the classifier for each (optional) fold.
        verbose
            Verbosity level during the training of the classifier.
            0 is silent, 1 a progress bar and 2 prints the statistics after each epoch.
        train_kwargs
            Optional additional kwargs when fitting the classifier.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            threshold=threshold,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            metric_fn=metric_fn,
            metric_name=metric_name,
            train_size=train_size,
            n_folds=n_folds,
            seed=seed,
            data_type=data_type
        )
        self.meta.update({'backend': 'tensorflow'})

        # define and compile classifier model
        self.model = model
        self.compile_kwargs = {'optimizer': optimizer(learning_rate=learning_rate), 'loss': BinaryCrossentropy()}
        if isinstance(compile_kwargs, dict):
            self.compile_kwargs.update(compile_kwargs)
        self.train_kwargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': verbose}
        if isinstance(train_kwargs, dict):
            self.train_kwargs.update(train_kwargs)

    def score(self, x: np.ndarray) -> float:
        """
        Compute the out-of-fold drift metric such as the accuracy from a classifier
        trained to distinguish the reference data from the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        Drift metric (e.g. accuracy) obtained from out-of-fold predictions from a trained classifier.
        """
        x_ref, x = self.preprocess(x)
        x, y, splits = self.get_splits(x_ref, x)

        # iterate over folds: train a new model for each fold and make out-of-fold (oof) predictions
        preds_oof, idx_oof = [], []
        for idx_tr, idx_te in splits:
            x_tr, y_tr, x_te = x[idx_tr], np.eye(2)[y[idx_tr]], x[idx_te]
            clf = tf.keras.models.clone_model(self.model)
            clf.compile(**self.compile_kwargs)
            clf.fit(x=x_tr, y=y_tr, **self.train_kwargs)
            preds = clf.predict(x_te, batch_size=self.train_kwargs['batch_size'])
            preds_oof.append(preds)
            idx_oof.append(idx_te)
        preds_oof = np.concatenate(preds_oof, axis=0)[:, 1]
        idx_oof = np.concatenate(idx_oof, axis=0)
        drift_metric = self.metric_fn(y[idx_oof], preds_oof)
        return drift_metric
