import numpy as np
import tensorflow as tf
from scipy import stats
from typing import Callable, Dict, Optional, Union, Tuple
from alibi_detect.cd.base import BaseNMEDrift


class NMEDriftTF(BaseNMEDrift):
    def __init__(
            self,
            x_ref: np.ndarray,
            kernel: Union[tf.keras.Model, tf.keras.Sequential],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            train_size: Optional[float] = .75,
            cov_reg: float = 1e-6,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam,
            learning_rate: float = 1e-3,
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
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            train_size=train_size,
            cov_reg=cov_reg,
            data_type=data_type
        )
        self.meta.update({'backend': 'tensorflow'})

        init_test_locations = tf.convert_to_tensor(self.init_test_locations(x_ref))
        self.nme_embedder = NMEDriftTF.NMEEmbedder(kernel, init_test_locations)

        self.train_kwargs = {
            'batch_size': batch_size,
            'epochs': epochs, 'verbose': verbose,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
        }
        if isinstance(train_kwargs, dict):
            self.train_kwargs.update(train_kwargs)

    class NMEEmbedder(tf.keras.Model):
        def __init__(self, kernel: tf.keras.Model, init_locations: tf.Tensor):
            super().__init__()
            self.kernel = kernel
            self.test_locs = tf.Variable(init_locations)

        def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
            x, y = inputs
            k_xtl = self.kernel(x, self.test_locs)
            k_ytl = self.kernel(y, self.test_locs)
            return k_xtl - k_ytl

    def score(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Compute the out-of-fold drift metric such as the accuracy from a classifier
        trained to distinguish the reference data from the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value, and a notion of distance between the trained classifier's out-of-fold performance
        and that which we'd expect under the null assumption of no drift.
        """
        x_ref, x = self.preprocess(x)
        (x_ref_train, x_ref_test), (x_train, x_test) = self.get_splits(x_ref, x)

        train_args = [self.nme_embedder, self.cov_reg, x_ref_train, x_train]
        self.trainer(*train_args, **self.train_kwargs)
        embeddings = NMEDriftTF.embed_batch(
            tf.convert_to_tensor(x_ref_test), tf.convert_to_tensor(x_test), 
            self.nme_embedder, self.train_kwargs['batch_size']
        )
        nme_estimate = NMEDriftTF.embedding_to_estimate(embeddings, cov_reg=self.cov_reg)
        new_test_locs = self.nme_embedder.test_locs.numpy()

        p_val = stats.chi2.sf(nme_estimate.numpy(), self.J)
        return p_val, nme_estimate, new_test_locs

    @staticmethod
    def embedding_to_estimate(z: tf.Tensor, cov_reg: float = 1e-12) -> tf.Tensor:
        n, J = z.shape
        S = tf.einsum('ij,ik->jk', (z - tf.reduce_mean(z, 0)), (z - tf.reduce_mean(z, 0)))/(n-1)
        S += cov_reg * tf.eye(J)
        S_inv = tf.linalg.inv(S)
        return n * tf.reduce_mean(z, 0)[None, :] @ S_inv @ tf.reduce_mean(z, 0)[:, None]

    @staticmethod
    def embed_batch(
        x_ref: tf.Tensor,
        x: tf.Tensor,
        nme_embedder: NMEEmbedder,
        batch_size: int
    ) -> tf.Tensor:
        n = len(x)
        n_minibatch = int(np.ceil(n / batch_size))
        embeddings = []
    
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_ref_batch, x_batch = x_ref[istart:istop], x[istart:istop]
            embeddings_batch = nme_embedder((x_ref_batch, x_batch), training=False)
            embeddings.append(embeddings_batch)
        return tf.concat(embeddings, 0)

    @staticmethod
    def trainer(
        nme_embedder: NMEEmbedder,
        cov_reg: float,
        x_ref_train: tf.Tensor,
        x_train: tf.Tensor,
        optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam,
        learning_rate: float=1e-3,
        epochs: int = 20,
        batch_size: int = 64,
        buffer_size: int = 1024,
        verbose: bool = True
    ) -> None:
       
        train_data = tf.data.Dataset.from_tensor_slices((x_ref_train, x_train))
        train_data = train_data.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
        n_minibatch = int(np.ceil(x_ref_train.shape[0] / batch_size))

        optim = optimizer(learning_rate)

        # iterate over epochs
        est_ma = 0.
        for epoch in range(epochs):
            if verbose:
                pbar = tf.keras.utils.Progbar(n_minibatch, 1)

            # iterate over the batches of the dataset
            for step, train_batch in enumerate(train_data):
                with tf.GradientTape() as tape:
                    embedding = nme_embedder(train_batch)
                    neg_estimate = -NMEDriftTF.embedding_to_estimate(embedding, cov_reg=cov_reg)
                grads = tape.gradient(neg_estimate, nme_embedder.trainable_weights)
                optim.apply_gradients(zip(grads, nme_embedder.trainable_weights))
                if verbose:
                    est_ma = est_ma + (-float(neg_estimate) - est_ma) / (step + 1)
                    pbar_values = [('estimate', est_ma)]
                    pbar.add(1, values=pbar_values)