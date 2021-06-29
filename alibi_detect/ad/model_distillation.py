import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import kld, categorical_crossentropy
from typing import Callable, Dict, Tuple, Union
from alibi_detect.models.tensorflow.trainer import trainer
from alibi_detect.models.tensorflow.losses import loss_distillation
from alibi_detect.utils.tensorflow.prediction import predict_batch
from alibi_detect.base import (BaseDetector, FitMixin, ThresholdMixin,
                               adversarial_prediction_dict)

logger = logging.getLogger(__name__)


class ModelDistillation(BaseDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 distilled_model: tf.keras.Model = None,
                 model: tf.keras.Model = None,
                 loss_type: str = 'kld',
                 temperature: float = 1.,
                 data_type: str = None
                 ) -> None:
        """
        Model distillation concept drift and adversarial detector.

        Parameters
        ----------
        threshold
            Threshold used for score to determine adversarial instances.
        distilled_model
            A tf.keras model to distill.
        model
            A trained tf.keras classification model.
        loss_type
            Loss for distillation. Supported: 'kld', 'xent'
        temperature
            Temperature used for model prediction scaling.
            Temperature <1 sharpens the prediction probability distribution.
        data_type
            Optionally specifiy the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.model = model
        for layer in self.model.layers:  # freeze model layers
            layer.trainable = False

        if isinstance(distilled_model, tf.keras.Model):
            self.distilled_model = distilled_model
        else:
            raise TypeError('No valid format detected for `distilled_model` (tf.keras.Model) ')
        self.loss_type = loss_type
        self.temperature = temperature

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type

    def fit(self,
            X: np.ndarray,
            loss_fn: tf.keras.losses = loss_distillation,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            epochs: int = 20,
            batch_size: int = 128,
            verbose: bool = True,
            log_metric: Tuple[str, "tf.keras.metrics"] = None,
            callbacks: tf.keras.callbacks = None,
            preprocess_fn: Callable = None
            ) -> None:
        """
        Train ModelDistillation detector.

        Parameters
        ----------
        X
            Training batch.
        loss_fn
            Loss function used for training.
        optimizer
            Optimizer used for training.
        epochs
            Number of training epochs.
        batch_size
            Batch size used for training.
        verbose
            Whether to print training progress.
        log_metric
            Additional metrics whose progress will be displayed if verbose equals True.
        callbacks
            Callbacks used during training.
        preprocess_fn
            Preprocessing function applied to each training batch.
        """
        # train arguments
        args = [self.distilled_model, loss_fn, X]
        kwargs = {
            'optimizer': optimizer,
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': verbose,
            'log_metric': log_metric,
            'callbacks': callbacks,
            'preprocess_fn': preprocess_fn,
            'loss_fn_kwargs': {
                'model': self.model,
                'loss_type': self.loss_type,
                'temperature': self.temperature
            }
        }

        # train
        trainer(*args, **kwargs)

    def infer_threshold(self,
                        X: np.ndarray,
                        threshold_perc: float = 99.,
                        margin: float = 0.,
                        batch_size: int = int(1e10)
                        ) -> None:
        """
        Update threshold by a value inferred from the percentage of instances considered to be
        adversarial in a sample of the dataset.

        Parameters
        ----------
        X
            Batch of instances.
        threshold_perc
            Percentage of X considered to be normal based on the adversarial score.
        margin
            Add margin to threshold. Useful if adversarial instances have significantly higher scores and there
            is no adversarial instance in X.
        batch_size
            Batch size used when computing scores.
        """
        # compute adversarial scores
        adv_score = self.score(X, batch_size=batch_size)

        # update threshold
        self.threshold = np.percentile(adv_score, threshold_perc) + margin

    def score(self, X: np.ndarray, batch_size: int = int(1e10), return_predictions: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute adversarial scores.

        Parameters
        ----------
        X
            Batch of instances to analyze.
        batch_size
            Batch size used when computing scores.
        return_predictions
            Whether to return the predictions of the classifier on the original and reconstructed instances.

        Returns
        -------
        Array with adversarial scores for each instance in the batch.
        """

        # model predictions
        y = predict_batch(X, self.model, batch_size=batch_size)
        y_distilled = predict_batch(X, self.distilled_model, batch_size=batch_size)

        # scale predictions
        if self.temperature != 1.:
            y = y ** (1 / self.temperature)  # type: ignore
            y = (y / tf.reshape(tf.reduce_sum(y, axis=-1), (-1, 1))).numpy()

        if self.loss_type == 'kld':
            score = kld(y, y_distilled).numpy()
        elif self.loss_type == 'xent':
            score = categorical_crossentropy(y, y_distilled).numpy()
        else:
            raise NotImplementedError

        if return_predictions:
            return score, y, y_distilled
        else:
            return score

    def predict(self, X: np.ndarray, batch_size: int = int(1e10), return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[str, np.ndarray]]:
        """
        Predict whether instances are adversarial instances or not.

        Parameters
        ----------
        X
            Batch of instances.
        batch_size
            Batch size used when computing scores.
        return_instance_score
            Whether to return instance level adversarial scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the adversarial predictions and instance level adversarial scores.
        """
        score = self.score(X, batch_size=batch_size)

        # values above threshold are adversarial
        pred = (score > self.threshold).astype(int)  # type: ignore

        # populate output dict
        ad = adversarial_prediction_dict()
        ad['meta'] = self.meta
        ad['data']['is_adversarial'] = pred
        if return_instance_score:
            ad['data']['instance_score'] = score
        return ad
