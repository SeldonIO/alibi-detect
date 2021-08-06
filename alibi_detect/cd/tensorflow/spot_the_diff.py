import numpy as np
import tensorflow as tf
from typing import Callable, Dict, Optional, Union
from alibi_detect.cd.tensorflow.classifier import ClassifierDriftTF
from alibi_detect.utils.tensorflow import GaussianRBF


class SpotTheDiffDriftTF:
    def __init__(
            self,
            x_ref: np.ndarray,
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            kernel: Callable = None,
            n_diffs: int = 1,
            l1_reg: float = 0.01,
            binarize_preds: bool = False,
            train_size: Optional[float] = .75,
            n_folds: Optional[int] = None,
            retrain_from_scratch: bool = True,
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
        Classifier-based drift detector with a classifier of form y = a + b_1*k(x,w_1) + ... + b_J*k(x,w_J),
        where k is a kernel and w_1,...,w_J are learnable test locations. If drift has occured the test locations
        learn to be more/less (given by sign of b_i) simmilar to test instances than reference instances.
        The test locations are regularised to be close to the average reference instance such that the difference
        is then interpretable as the transformation required to make the average instance more/less like a test instance
        than a reference instance.

        The classifier is trained on a fraction of the combined reference and test data and drift is detected on
        the remaining data. To use all the data to detect drift, a stratified cross-validation scheme can be chosen.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        model
            TensorFlow classification model used for drift detection.
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
        kernel
            Kernel used to define simmilarity between instances, defaults to Gaussian RBF
        n_diffs
            The number of test locations to use, each corresponding to an interpretable difference.
        l1_reg
            Strength of l1 regularisation to apply to the differences.
        binarize_preds
            Whether to test for discrepency on soft (e.g. prob/log-prob) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
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

        if kernel is None:
            kernel = GaussianRBF(trainable=True)  # TODO: Think about

        model = SpotTheDiffDriftTF.InterpretableClf(kernel, x_ref, n_diffs=n_diffs)
        reg_loss_fn = (lambda model: tf.reduce_mean(tf.abs(model.diffs)) * l1_reg)

        self._detector = ClassifierDriftTF(
            x_ref=x_ref,
            model=model,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
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
            epochs=epochs,
            verbose=verbose,
            train_kwargs=train_kwargs,
            data_type=data_type
        )
        self.meta = self._detector.meta
        self.meta['name'] = 'SpotTheDiffDrift'

    class InterpretableClf(tf.keras.Model):
        def __init__(self, kernel: tf.keras.Model, x_ref: np.ndarray, n_diffs: int = 1):
            super().__init__()
            self.config = {'kernel': kernel, 'x_ref': x_ref, 'n_diffs': n_diffs}
            x_ref = tf.convert_to_tensor(x_ref)
            self.kernel = kernel
            self.mean = tf.reduce_mean(x_ref, 0)
            # TODO: Initialisation here is important. What is best way?
            self.diffs = tf.Variable(
                tf.random.normal((n_diffs,) + x_ref.shape[1:]) * tf.math.reduce_std(x_ref, 0)
            )
            self.bias = tf.Variable(tf.zeros((1,)))
            self.coeffs = tf.Variable(tf.zeros((n_diffs,)))

        def call(self, x: tf.Tensor) -> tf.Tensor:
            k_xtl = self.kernel(x, self.mean + self.diffs)
            logits = self.bias + k_xtl @ self.coeffs[:, None]  # exp ensures coeff>0 for interpretability
            return tf.concat([-logits, logits], axis=-1)

        def get_config(self):  # not needed for sequential/functional API models
            return self.config

        @classmethod
        def from_config(cls, config):  # not needed for sequential/functional API models
            return cls(**config)

    def predict(
        self, x: np.ndarray,  return_p_val: bool = True, return_distance: bool = True,
        return_probs: bool = True, return_model: bool = False
    ) -> Dict[Dict[str, str], Dict[str, Union[int, float, Callable]]]:
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
        'data' contains the drift prediction and optionally the p-value and performance of
        the classifier relative to its expectation under the no-change null.
        """
        preds = self._detector.predict(x, return_p_val, return_distance, return_probs, return_model=True)
        preds['data']['diffs'] = preds['data']['model'].diffs.numpy()
        preds['data']['diff_coeffs'] = preds['data']['model'].coeffs.numpy()
        if not return_model:
            del preds['data']['model']
        return preds
