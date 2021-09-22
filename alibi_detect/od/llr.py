from functools import partial
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow_probability.python.distributions.distribution import Distribution
from typing import Callable, Dict, Tuple, Union
from alibi_detect.models.tensorflow import PixelCNN
from alibi_detect.models.tensorflow.trainer import trainer
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, outlier_prediction_dict
from alibi_detect.utils.tensorflow.prediction import predict_batch
from alibi_detect.utils.perturbation import mutate_categorical

logger = logging.getLogger(__name__)


def build_model(dist: Union[Distribution, PixelCNN], input_shape: tuple = None, filepath: str = None) \
        -> Tuple[tf.keras.Model, Union[Distribution, PixelCNN]]:
    """
    Create tf.keras.Model from TF distribution.

    Parameters
    ----------
    dist
        TensorFlow distribution.
    input_shape
        Input shape of the model.
    filepath
        File to load model weights from.

    Returns
    -------
    TensorFlow model.
    """
    x_in = Input(shape=input_shape)
    log_prob = dist.log_prob(x_in)
    model = Model(inputs=x_in, outputs=log_prob)
    model.add_loss(-tf.reduce_mean(log_prob))
    if isinstance(filepath, str):
        model.load_weights(filepath)
    return model, dist


class LLR(BaseDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 model: Union[tf.keras.Model, Distribution, PixelCNN] = None,
                 model_background: Union[tf.keras.Model, Distribution, PixelCNN] = None,
                 log_prob: Callable = None,
                 sequential: bool = False,
                 data_type: str = None
                 ) -> None:
        """
        Likelihood Ratios for Out-of-Distribution Detection. Ren, J. et al. NeurIPS 2019.
        https://arxiv.org/abs/1906.02845

        Parameters
        ----------
        threshold
            Threshold used for the likelihood ratio (LLR) to determine outliers.
        model
            Generative model, defaults to PixelCNN.
        model_background
            Optional model for the background. Only needed if it is different from `model`.
        log_prob
            Function used to evaluate log probabilities under the model
            if the model does not have a `log_prob` function.
        sequential
            Whether the data is sequential. Used to create targets during training.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.has_log_prob = True if hasattr(model, 'log_prob') else False
        self.sequential = sequential
        self.log_prob = log_prob
        self.threshold = threshold

        # semantic model trained on original data
        self.dist_s = model
        # background model trained on perturbed data
        if model_background is None:
            try:
                self.dist_b = model.copy()
            except AttributeError:
                self.dist_b = tf.keras.models.clone_model(model)
        else:
            self.dist_b = model_background

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type

    def fit(self,
            X: np.ndarray,
            mutate_fn: Callable = mutate_categorical,
            mutate_fn_kwargs: dict = {'rate': .2, 'seed': 0, 'feature_range': (0, 255)},
            mutate_batch_size: int = int(1e10),
            loss_fn: tf.keras.losses = None,
            loss_fn_kwargs: dict = None,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            epochs: int = 20,
            batch_size: int = 64,
            verbose: bool = True,
            log_metric: Tuple[str, "tf.keras.metrics"] = None,
            callbacks: tf.keras.callbacks = None
            ) -> None:
        """
        Train semantic and background generative models.

        Parameters
        ----------
        X
            Training batch.
        mutate_fn
            Mutation function used to generate the background dataset.
        mutate_fn_kwargs
            Kwargs for the mutation function used to generate the background dataset.
            Default values set for an image dataset.
        mutate_batch_size
            Batch size used to generate the mutations for the background dataset.
        loss_fn
            Loss function used for training.
        loss_fn_kwargs
            Kwargs for loss function.
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
        """
        input_shape = X.shape[1:]

        # training arguments
        kwargs = {'epochs': epochs,
                  'batch_size': batch_size,
                  'verbose': verbose,
                  'callbacks': callbacks}

        # create background data
        mutate_fn = partial(mutate_fn, **mutate_fn_kwargs)
        X_back = predict_batch(X, mutate_fn, batch_size=mutate_batch_size, dtype=X.dtype)

        # prepare sequential data
        if self.sequential and not self.has_log_prob:
            y, y_back = X[:, 1:], X_back[:, 1:]  # type: ignore
            X, X_back = X[:, :-1], X_back[:, :-1]  # type: ignore
        else:
            y, y_back = None, None

        # check if model needs to be built
        use_build = True if self.has_log_prob and not isinstance(self.dist_s, tf.keras.Model) else False

        if use_build:
            # build and train semantic model
            self.model_s = build_model(self.dist_s, input_shape)[0]
            self.model_s.compile(optimizer=optimizer)
            self.model_s.fit(X, **kwargs)
            # build and train background model
            self.model_b = build_model(self.dist_b, input_shape)[0]
            self.model_b.compile(optimizer=optimizer)
            self.model_b.fit(X_back, **kwargs)
        else:
            # update training arguments
            kwargs.update({
                'optimizer': optimizer,
                'loss_fn_kwargs': loss_fn_kwargs,
                'log_metric': log_metric
            })

            # train semantic model
            args = [self.dist_s, loss_fn, X]
            kwargs.update({'y_train': y})
            trainer(*args, **kwargs)

            # train background model
            args = [self.dist_b, loss_fn, X_back]
            kwargs.update({'y_train': y_back})
            trainer(*args, **kwargs)

    def infer_threshold(self,
                        X: np.ndarray,
                        outlier_type: str = 'instance',
                        threshold_perc: float = 95.,
                        batch_size: int = int(1e10)
                        ) -> None:
        """
        Update LLR threshold by a value inferred from the percentage of instances
        considered to be outliers in a sample of the dataset.

        Parameters
        ----------
        X
            Batch of instances.
        outlier_type
            Predict outliers at the 'feature' or 'instance' level.
        threshold_perc
            Percentage of sorted feature level outlier scores used to predict instance level outlier.
        batch_size
            Batch size for the generative model evaluations.
        """
        # compute outlier scores
        fscore, iscore = self.score(X, batch_size=batch_size)
        if outlier_type == 'feature':
            outlier_score = fscore
        elif outlier_type == 'instance':
            outlier_score = iscore
        else:
            raise ValueError('`outlier_score` needs to be either `feature` or `instance`.')

        # update threshold
        self.threshold = np.percentile(outlier_score, threshold_perc)

    def logp(self, dist, X: np.ndarray, return_per_feature: bool = False, batch_size: int = int(1e10)) \
            -> np.ndarray:
        """
        Compute log probability of a batch of instances under the generative model.

        Parameters
        ----------
        dist
            Distribution of the model.
        X
            Batch of instances.
        return_per_feature
            Return log probability per feature.
        batch_size
            Batch size for the generative model evaluations.

        Returns
        -------
        Log probabilities.
        """
        logp_fn = partial(dist.log_prob, return_per_feature=return_per_feature)
        return predict_batch(X, logp_fn, batch_size=batch_size)

    def logp_alt(self, model: tf.keras.Model, X: np.ndarray, return_per_feature: bool = False,
                 batch_size: int = int(1e10)) -> np.ndarray:
        """
        Compute log probability of a batch of instances using the log_prob function
        defined by the user.

        Parameters
        ----------
        model
            Trained model.
        X
            Batch of instances.
        return_per_feature
            Return log probability per feature.
        batch_size
            Batch size for the generative model evaluations.

        Returns
        -------
        Log probabilities.
        """
        if self.sequential:
            y, X = X[:, 1:], X[:, :-1]
        else:
            y = X.copy()
        y_preds = predict_batch(X, model, batch_size=batch_size)
        logp = self.log_prob(y, y_preds).numpy()
        if return_per_feature:
            return logp
        else:
            axis = tuple(np.arange(len(logp.shape))[1:])
            return np.mean(logp, axis=axis)

    def llr(self, X: np.ndarray, return_per_feature: bool, batch_size: int = int(1e10)) -> np.ndarray:
        """
        Compute likelihood ratios.

        Parameters
        ----------
        X
            Batch of instances.
        return_per_feature
            Return likelihood ratio per feature.
        batch_size
            Batch size for the generative model evaluations.

        Returns
        -------
        Likelihood ratios.
        """
        logp_fn = self.logp if not isinstance(self.log_prob, Callable) else self.logp_alt  # type: ignore
        logp_s = logp_fn(self.dist_s, X, return_per_feature=return_per_feature, batch_size=batch_size)
        logp_b = logp_fn(self.dist_b, X, return_per_feature=return_per_feature, batch_size=batch_size)
        return logp_s - logp_b

    def feature_score(self, X: np.ndarray, batch_size: int = int(1e10)) -> np.ndarray:
        """ Feature-level negative likelihood ratios. """
        return - self.llr(X, True, batch_size=batch_size)

    def instance_score(self, X: np.ndarray, batch_size: int = int(1e10)) -> np.ndarray:
        """ Instance-level negative likelihood ratios. """
        return - self.llr(X, False, batch_size=batch_size)

    def score(self, X: np.ndarray, batch_size: int = int(1e10)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Feature-level and instance-level outlier scores.
        The scores are equal to the negative likelihood ratios.
        """
        fscore = self.feature_score(X, batch_size=batch_size)
        iscore = self.instance_score(X, batch_size=batch_size)
        return fscore, iscore

    def predict(self,
                X: np.ndarray,
                outlier_type: str = 'instance',
                batch_size: int = int(1e10),
                return_feature_score: bool = True,
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Predict whether instances are outliers or not.

        Parameters
        ----------
        X
            Batch of instances.
        outlier_type
            Predict outliers at the 'feature' or 'instance' level.
        batch_size
            Batch size used when making predictions with the generative model.
        return_feature_score
            Whether to return feature level outlier scores.
        return_instance_score
            Whether to return instance level outlier scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the outlier predictions and both feature and instance level outlier scores.
        """
        # compute outlier scores
        fscore, iscore = self.score(X, batch_size=batch_size)
        if outlier_type == 'feature':
            outlier_score = fscore
        elif outlier_type == 'instance':
            outlier_score = iscore
        else:
            raise ValueError('`outlier_score` needs to be either `feature` or `instance`.')

        # values above threshold are outliers
        outlier_pred = (outlier_score > self.threshold).astype(int)

        # populate output dict
        od = outlier_prediction_dict()
        od['meta'] = self.meta
        od['data']['is_outlier'] = outlier_pred
        if return_feature_score:
            od['data']['feature_score'] = fscore
        if return_instance_score:
            od['data']['instance_score'] = iscore
        return od
