from functools import partial
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, Optional, Tuple, Union
from alibi_detect.cd.base import BaseLearnedKernelDrift
from alibi_detect.utils.tensorflow.data import TFDataset
from alibi_detect.utils.tensorflow.misc import clone_model
from alibi_detect.utils.tensorflow.distance import mmd2_from_kernel_matrix, batch_compute_kernel_matrix


class LearnedKernelDriftTF(BaseLearnedKernelDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            kernel: tf.keras.Model,
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            n_permutations: int = 100,
            var_reg: float = 1e-5,
            reg_loss_fn: Callable = (lambda kernel: 0),
            train_size: Optional[float] = .75,
            retrain_from_scratch: bool = True,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            preprocess_batch_fn: Optional[Callable] = None,
            epochs: int = 3,
            verbose: int = 0,
            train_kwargs: Optional[dict] = None,
            dataset: Callable = TFDataset,
            data_type: Optional[str] = None
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) data drift detector where the kernel is trained to maximise an
        estimate of the test power. The kernel is trained on a split of the reference and test instances
        and then the MMD is evaluated on held out instances and a permutation test is performed.

        For details see Liu et al (2020): Learning Deep Kernels for Non-Parametric Two-Sample Tests
        (https://arxiv.org/abs/2002.09116)


        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        kernel
            Trainable TensorFlow model that returns a similarity between two instances.
        p_val
            p-value used for the significance of the test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before applying the kernel.
        n_permutations
            The number of permutations to use in the permutation test once the MMD has been computed.
        var_reg
            Constant added to the estimated variance of the MMD for stability.
        reg_loss_fn
            The regularisation term reg_loss_fn(kernel) is added to the loss function being optimized.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the kernel.
            The drift is detected on `1 - train_size`.
        retrain_from_scratch
            Whether the kernel should be retrained from scratch for each set of test data or whether
            it should instead continue training from where it left off on the previous set.
        optimizer
            Optimizer used during training of the kernel.
        learning_rate
            Learning rate used by optimizer.
        batch_size
            Batch size used during training of the kernel.
        preprocess_batch_fn
            Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
            processed by the kernel.
        epochs
            Number of training epochs for the kernel. Corresponds to the smaller of the reference and test sets.
        verbose
            Verbosity level during the training of the kernel. 0 is silent, 1 a progress bar.
        train_kwargs
            Optional additional kwargs when training the kernel.
        dataset
            Dataset object used during training.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            n_permutations=n_permutations,
            train_size=train_size,
            retrain_from_scratch=retrain_from_scratch,
            data_type=data_type
        )
        self.meta.update({'backend': 'tensorflow'})

        # define and compile kernel
        self.original_kernel = kernel
        self.kernel = clone_model(kernel)

        self.dataset = partial(dataset, batch_size=batch_size, shuffle=True)
        self.kernel_mat_fn = partial(
            batch_compute_kernel_matrix, preprocess_fn=preprocess_batch_fn, batch_size=batch_size
        )
        self.train_kwargs = {'optimizer': optimizer, 'epochs': epochs, 'learning_rate': learning_rate,
                             'reg_loss_fn': reg_loss_fn, 'preprocess_fn': preprocess_batch_fn, 'verbose': verbose}
        if isinstance(train_kwargs, dict):
            self.train_kwargs.update(train_kwargs)

        self.j_hat = LearnedKernelDriftTF.JHat(self.kernel, var_reg)

    class JHat(tf.keras.Model):
        """
        A module that wraps around the kernel. When passed a batch of reference and batch of test
        instances it returns an estimate of a correlate of test power.
        Equation 4 of https://arxiv.org/abs/2002.09116
        """
        def __init__(self, kernel: tf.keras.Model, var_reg: float):
            super().__init__()
            self.config = {'kernel': kernel, 'var_reg': var_reg}
            self.kernel = kernel
            self.var_reg = var_reg

        def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            k_xx, k_yy, k_xy = self.kernel(x, x), self.kernel(y, y), self.kernel(x, y)
            h_mat = k_xx + k_yy - k_xy - tf.transpose(k_xy)

            n = len(x)
            mmd2_est = (tf.reduce_sum(h_mat)-tf.linalg.trace(h_mat))/(n*(n-1))
            var_est = (4*tf.reduce_sum(tf.reduce_sum(h_mat, axis=-1)**2)/(n**3) -
                       4*tf.reduce_sum(h_mat)**2/(n**4))
            reg_var_est = var_est + self.var_reg

            return mmd2_est/tf.math.sqrt(reg_var_est)

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, np.ndarray]:
        """
        Compute the p-value resulting from a permutation test using the maximum mean discrepancy
        as a distance measure between the reference data and the data to be tested. The kernel
        used within the MMD is first trained to maximise an estimate of the resulting test power.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the MMD^2 between the reference and test set
        and the MMD^2 values from the permutation test.
        """
        x_ref, x_cur = self.preprocess(x)
        (x_ref_tr, x_cur_tr), (x_ref_te, x_cur_te) = self.get_splits(x_ref, x_cur)
        ds_ref_tr, ds_cur_tr = self.dataset(x_ref_tr), self.dataset(x_cur_tr)

        self.kernel = clone_model(self.original_kernel) if self.retrain_from_scratch else self.kernel
        train_args = [self.j_hat, (ds_ref_tr, ds_cur_tr)]
        LearnedKernelDriftTF.trainer(*train_args, **self.train_kwargs)  # type: ignore

        if isinstance(x_ref_te, np.ndarray) and isinstance(x_cur_te, np.ndarray):
            x_all = np.concatenate([x_ref_te, x_cur_te], axis=0)
        else:
            x_all = x_ref_te + x_cur_te
        kernel_mat = self.kernel_mat_fn(x_all, x_all, self.kernel)
        kernel_mat = kernel_mat - tf.linalg.diag(tf.linalg.diag_part(kernel_mat))  # zero diagonal
        mmd2 = mmd2_from_kernel_matrix(kernel_mat, len(x_cur_te), permute=False, zero_diag=False).numpy()
        mmd2_permuted = np.array(
            [mmd2_from_kernel_matrix(kernel_mat, len(x_cur_te), permute=True, zero_diag=False).numpy()
                for _ in range(self.n_permutations)]
        )
        p_val = (mmd2 <= mmd2_permuted).mean()
        return p_val, mmd2, mmd2_permuted

    @staticmethod
    def trainer(
        j_hat: JHat,
        datasets: Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence],
        optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam,
        learning_rate: float = 1e-3,
        preprocess_fn: Callable = None,
        epochs: int = 20,
        reg_loss_fn: Callable = (lambda kernel: 0),
        verbose: int = 1,
    ) -> None:
        """
        Train the kernel to maximise an estimate of test power using minibatch gradient descent.
        """
        ds_ref, ds_cur = datasets
        optimizer = optimizer(learning_rate)
        n_minibatch = min(len(ds_ref), len(ds_cur))
        # iterate over epochs
        loss_ma = 0.
        for epoch in range(epochs):
            if verbose:
                pbar = tf.keras.utils.Progbar(n_minibatch, 1)
            for step, (x_ref, x_cur) in enumerate(zip(ds_ref, ds_cur)):
                if isinstance(preprocess_fn, Callable):  # type: ignore
                    x_ref, x_cur = preprocess_fn(x_ref), preprocess_fn(x_cur)
                with tf.GradientTape() as tape:
                    estimate = j_hat(x_ref, x_cur)
                    loss = -estimate + reg_loss_fn(j_hat.kernel)  # ascent
                grads = tape.gradient(loss, j_hat.trainable_weights)
                optimizer.apply_gradients(zip(grads, j_hat.trainable_weights))
                if verbose == 1:
                    loss_ma = loss_ma + (loss - loss_ma) / (step + 1)
                    pbar_values = [('loss', loss_ma)]
                    pbar.add(1, values=pbar_values)
