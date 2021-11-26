import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from typing import Dict, Tuple, Union
from alibi_detect.models.tensorflow.autoencoder import Seq2Seq, EncoderLSTM, DecoderLSTM
from alibi_detect.models.tensorflow.trainer import trainer
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, outlier_prediction_dict
from alibi_detect.utils.tensorflow.prediction import predict_batch

logger = logging.getLogger(__name__)


class OutlierSeq2Seq(BaseDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 n_features: int,
                 seq_len: int,
                 threshold: Union[float, np.ndarray] = None,
                 seq2seq: tf.keras.Model = None,
                 threshold_net: tf.keras.Model = None,
                 latent_dim: int = None,
                 output_activation: str = None,
                 beta: float = 1.
                 ) -> None:
        """
        Seq2Seq-based outlier detector.

        Parameters
        ----------
        n_features
            Number of features in the time series.
        seq_len
            Sequence length fed into the Seq2Seq model.
        threshold
            Threshold used for outlier detection. Can be a float or feature-wise array.
        seq2seq
            A trained seq2seq model if available.
        threshold_net
            Layers for the threshold estimation network wrapped in a
            tf.keras.Sequential class if no 'seq2seq' is specified.
        latent_dim
            Latent dimension of the encoder and decoder.
        output_activation
            Activation used in the Dense output layer of the decoder.
        beta
            Weight on the threshold estimation loss term.
        """
        super().__init__()

        if threshold is None:
            threshold = 0.
            logger.warning('No explicit threshold level set. Threshold defaults to 0. '
                           'A threshold can be inferred using `infer_threshold`.')

        self.threshold = threshold
        self.shape = (-1, seq_len, n_features)
        self.latent_dim = (latent_dim // 2) * 2
        if self.latent_dim != latent_dim:
            logger.warning('Odd values for `latent_dim` are not supported, because '
                           'of Bidirectional(LSTM(latent_dim // 2,...) in the encoder. '
                           f'{self.latent_dim} is used instead of {latent_dim}.)')

        self.output_activation = output_activation

        if threshold_net is None and seq2seq is None:  # default threshold network
            threshold_net = tf.keras.Sequential(
                [
                    InputLayer(input_shape=(seq_len, self.latent_dim)),
                    Dense(64, activation=tf.nn.relu),
                    Dense(64, activation=tf.nn.relu),
                ])

        # check if model can be loaded, otherwise initialize a Seq2Seq model
        if isinstance(seq2seq, tf.keras.Model):
            self.seq2seq = seq2seq
        elif isinstance(latent_dim, int) and isinstance(threshold_net, tf.keras.Sequential):
            encoder_net = EncoderLSTM(self.latent_dim)
            decoder_net = DecoderLSTM(self.latent_dim, n_features, output_activation)
            self.seq2seq = Seq2Seq(encoder_net, decoder_net, threshold_net, n_features, beta=beta)
        else:
            raise TypeError('No valid format detected for `seq2seq` (tf.keras.Model), '
                            '`latent_dim` (int) or `threshold_net` (tf.keras.Sequential)')

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = 'time-series'

    def fit(self,
            X: np.ndarray,
            loss_fn: tf.keras.losses = tf.keras.losses.mse,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            epochs: int = 20,
            batch_size: int = 64,
            verbose: bool = True,
            log_metric: Tuple[str, "tf.keras.metrics"] = None,
            callbacks: tf.keras.callbacks = None,
            ) -> None:
        """
        Train Seq2Seq model.

        Parameters
        ----------
        X
            Univariate or multivariate time series.
            Shape equals (batch, features) or (batch, sequence length, features).
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
        """
        # targets for teacher-forcing
        if len(X.shape) == 2:
            y = np.roll(X, -1, axis=0).reshape(self.shape)
            X = X.reshape(self.shape)
        else:
            y = np.roll(X.reshape((-1, self.shape[-1])), -1, axis=0).reshape(self.shape)

        # train arguments
        args = [self.seq2seq, loss_fn, X]
        kwargs = {'y_train': y,
                  'optimizer': optimizer,
                  'epochs': epochs,
                  'batch_size': batch_size,
                  'verbose': verbose,
                  'log_metric': log_metric,
                  'callbacks': callbacks}

        # train
        trainer(*args, **kwargs)

    def infer_threshold(self,
                        X: np.ndarray,
                        outlier_perc: Union[int, float] = 100.,
                        threshold_perc: Union[int, float, np.ndarray, list] = 95.,
                        batch_size: int = int(1e10)
                        ) -> None:
        """
        Update the outlier threshold by using a sequence of instances from the dataset
        of which the fraction of features which are outliers are known. This fraction can be across
        all features or per feature.

        Parameters
        ----------
        X
            Univariate or multivariate time series.
        outlier_perc
            Percentage of sorted feature level outlier scores used to predict instance level outlier.
        threshold_perc
            Percentage of X considered to be normal based on the outlier score.
            Overall (float) or feature-wise (array or list).
        batch_size
            Batch size used when making predictions with the seq2seq model.
        """
        orig_shape = X.shape
        threshold_shape = (1, orig_shape[-1])
        if len(orig_shape) == 3:  # (batch_size, seq_len, n_features)
            threshold_shape = (1,) + threshold_shape  # type: ignore

        # compute outlier scores
        fscore, iscore = self.score(X, outlier_perc=outlier_perc, batch_size=batch_size)
        if outlier_perc == 100.:
            fscore = fscore.reshape((-1, self.shape[-1]))

        # update threshold
        if isinstance(threshold_perc, (int, float)) and outlier_perc == 100.:
            self.threshold += np.percentile(fscore, threshold_perc)
        elif isinstance(threshold_perc, (int, float)) and outlier_perc < 100.:
            self.threshold += np.percentile(iscore, threshold_perc)
        elif isinstance(threshold_perc, (list, np.ndarray)) and outlier_perc == 100.:
            self.threshold += np.diag(np.percentile(fscore, threshold_perc, axis=0)).reshape(threshold_shape)
        elif isinstance(threshold_perc, (list, np.ndarray)) and outlier_perc < 100.:
            # number feature scores used for outlier score
            n_score = int(np.ceil(.01 * outlier_perc * fscore.shape[1]))
            # compute threshold level by feature
            sorted_fscore = np.sort(fscore, axis=1)
            if len(orig_shape) == 3:  # (batch_size, seq_len, n_features)
                sorted_fscore_perc = sorted_fscore[:, -n_score:, :]  # (batch_size, n_score, n_features)
                self.threshold += np.mean(sorted_fscore_perc, axis=(0, 1)).reshape(threshold_shape)  # (1,1,n_features)
            else:  # (batch_size, n_features)
                sorted_fscore_perc = sorted_fscore[:, -n_score:]  # (batch_size, n_score)
                self.threshold += np.mean(sorted_fscore_perc, axis=0)  # float
        else:
            raise TypeError('Incorrect type for `threshold` and/or `threshold_perc`.')

    def feature_score(self, X_orig: np.ndarray, X_recon: np.ndarray, threshold_est: np.ndarray) -> np.ndarray:
        """
        Compute feature level outlier scores.

        Parameters
        ----------
        X_orig
            Original time series.
        X_recon
            Reconstructed time series.
        threshold_est
            Estimated threshold from the decoder's latent space.

        Returns
        -------
        Feature level outlier scores. Scores above 0 are outliers.
        """
        fscore = (X_orig - X_recon) ** 2
        # TODO: check casting if nb of features equals time dimension
        fscore_adj = fscore - threshold_est - self.threshold
        return fscore_adj

    def instance_score(self, fscore: np.ndarray, outlier_perc: float = 100.) -> np.ndarray:
        """
        Compute instance level outlier scores. `instance` in this case means the data along the
        first axis of the original time series passed to the predictor.

        Parameters
        ----------
        fscore
            Feature level outlier scores.
        outlier_perc
            Percentage of sorted feature level outlier scores used to predict instance level outlier.

        Returns
        -------
        Instance level outlier scores.
        """
        fscore_flat = fscore.reshape(fscore.shape[0], -1).copy()
        n_score_features = int(np.ceil(.01 * outlier_perc * fscore_flat.shape[1]))
        sorted_fscore = np.sort(fscore_flat, axis=1)
        sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
        iscore = np.mean(sorted_fscore_perc, axis=1)
        return iscore

    def score(self, X: np.ndarray, outlier_perc: float = 100., batch_size: int = int(1e10)) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute feature and instance level outlier scores.

        Parameters
        ----------
        X
            Univariate or multivariate time series.
        outlier_perc
            Percentage of sorted feature level outlier scores used to predict instance level outlier.
        batch_size
            Batch size used when making predictions with the seq2seq model.

        Returns
        -------
        Feature and instance level outlier scores.
        """
        # use the seq2seq model to reconstruct instances
        orig_shape = X.shape
        if len(orig_shape) == 2:
            X = X.reshape(self.shape)
        X_recon, threshold_est = predict_batch(X, self.seq2seq.decode_seq, batch_size=batch_size)

        if len(orig_shape) == 2:  # reshape back to original shape
            X = X.reshape(orig_shape)
            X_recon = X_recon.reshape(orig_shape)
            threshold_est = threshold_est.reshape(orig_shape)

        # compute feature and instance level scores
        fscore = self.feature_score(X, X_recon, threshold_est)
        iscore = self.instance_score(fscore, outlier_perc=outlier_perc)
        return fscore, iscore

    def predict(self,
                X: np.ndarray,
                outlier_type: str = 'instance',
                outlier_perc: float = 100.,
                batch_size: int = int(1e10),
                return_feature_score: bool = True,
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Compute outlier scores and transform into outlier predictions.

        Parameters
        ----------
        X
            Univariate or multivariate time series.
        outlier_type
            Predict outliers at the 'feature' or 'instance' level.
        outlier_perc
            Percentage of sorted feature level outlier scores used to predict instance level outlier.
        batch_size
            Batch size used when making predictions with the seq2seq model.
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
        fscore, iscore = self.score(X, outlier_perc=outlier_perc, batch_size=batch_size)
        if outlier_type == 'feature':
            outlier_score = fscore
        elif outlier_type == 'instance':
            outlier_score = iscore
        else:
            raise ValueError('`outlier_score` needs to be either `feature` or `instance`.')

        # values above threshold are outliers
        outlier_pred = (outlier_score > 0).astype(int)

        # populate output dict
        od = outlier_prediction_dict()
        od['meta'] = self.meta
        od['data']['is_outlier'] = outlier_pred
        if return_feature_score:
            od['data']['feature_score'] = fscore
        if return_instance_score:
            od['data']['instance_score'] = iscore
        return od
