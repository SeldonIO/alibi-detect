import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable, Dict, Optional, Tuple, Union, List
from alibi_detect.cd.base import BaseContextAwareDrift
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ContextAwareDriftTF(BaseContextAwareDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            c_ref: np.ndarray,
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            x_kernel: Callable = None,
            c_kernel: Callable = None,
            n_permutations: int = 1000,
            cond_prop: float = 0.25,
            lams: Optional[Tuple[float, float]] = None,
            batch_size: Optional[int] = 256,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
            verbose: bool = False
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) based context aware drift detector.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        c_ref
            Data used as context for the reference distribution.
        p_val
            p-value used for the significance of the permutation test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_kernel
            Kernel defined on the input data, defaults to Gaussian RBF kernel.
        c_kernel
            Kernel defined on the context data, defaults to Gaussian RBF kernel.
        n_permutations
            Number of permutations used in the permutation test.
        cond_prop
            Proportion of contexts held out to condition on.
        lams
            Ref and test regularisation parameters. Tuned if None.
        batch_size
            If not None, then compute batches of MMDs at a time (rather than all at once).
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        verbose
            Whether or not to print progress during configuration.
        """
        super().__init__(
            x_ref=x_ref,
            c_ref=c_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            x_kernel=x_kernel,
            c_kernel=c_kernel,
            n_permutations=n_permutations,
            cond_prop=cond_prop,
            lams=lams,
            batch_size=batch_size,
            input_shape=input_shape,
            data_type=data_type,
            verbose=verbose
        )
        self.meta.update({'backend': 'tensorflow'})

    def score(self,  # type: ignore[override]
              x: Union[np.ndarray, list], c: np.ndarray) -> Tuple[float, float, np.ndarray, Tuple]:
        """
        Compute the MMD based conditional test statistic, and perform a conditional permutation test to obtain a
        p-value representing the test statistic's extremity under the null hypothesis.

        Parameters
        ----------
        x
            Batch of instances.
        c
            Context associated with batch of instances.

        Returns
        -------
        p-value obtained from the conditional permutation test, the MMD-ADiTT test statistic, the permuted
        test statistics, and a tuple containing the coupling matrices (xx, yy, xy).
        """
        x_ref, x = self.preprocess(x)

        # Hold out a portion of contexts for conditioning on
        n, n_held = len(c), int(len(c)*self.cond_prop)
        inds_held = np.random.choice(n, n_held, replace=False)
        inds_test = np.setdiff1d(np.arange(n), inds_held)
        c_held = c[inds_held]
        c, x = c[inds_test], x[inds_test]
        n_ref, n_test = self.n, len(x)
        bools = tf.concat([tf.zeros(n_ref), tf.ones(n_test)], axis=0)

        # Compute kernel matrices
        x_all = tf.concat([x_ref, x], axis=0)
        c_all = tf.concat([self.c_ref, c], axis=0)
        K = self.x_kernel(x_all, x_all)
        L = self.c_kernel(c_all, c_all)
        L_held = self.c_kernel(c_held, c_all)

        # Fit and calibrate the domain classifier
        c_all_np, bools_np = c_all.numpy(), bools.numpy()
        self.clf.fit(c_all_np, bools_np)
        self.clf.calibrate(c_all_np, bools_np)

        # Obtain n_permutations conditional reassignments
        prop_scores = self.clf.predict(c_all_np)
        self.redrawn_bools = [tfp.distributions.Bernoulli(probs=prop_scores).sample()
                              for _ in range(self.n_permutations)]
        iters = tqdm(self.redrawn_bools, total=self.n_permutations) if self.verbose else self.redrawn_bools

        # Compute test stat on original and reassigned data
        stat, coupling_xx, coupling_yy, coupling_xy = self._cmmd(K, L, bools, L_held=L_held)
        permuted_stats = tf.stack([self._cmmd(K, L, perm_bools, L_held=L_held)[0] for perm_bools in iters])

        # Compute p-value
        p_val = tf.reduce_mean(tf.cast(stat <= permuted_stats, float))
        coupling = (coupling_xx.numpy(), coupling_yy.numpy(), coupling_xy.numpy())

        return p_val.numpy().item(), stat.numpy().item(), permuted_stats.numpy(), coupling

    def _cmmd(self, K: tf.Tensor, L: tf.Tensor, bools: tf.Tensor, L_held: tf.Tensor = None) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Private method to compute the MMD-ADiTT test statistic.
        """
        # Get ref/test indices
        idx_0, idx_1 = np.where(bools == 0)[0], np.where(bools == 1)[0]
        n_ref, n_test = len(idx_0), len(idx_1)

        # Form kernel matrices
        L_0, L_1 = tf.gather(tf.gather(L, idx_0), idx_0, axis=1), tf.gather(tf.gather(L, idx_1), idx_1, axis=1)
        K_0, K_1 = tf.gather(tf.gather(K, idx_0), idx_0, axis=1), tf.gather(tf.gather(K, idx_1), idx_1, axis=1)
        # Avoid using tf.gather_nd since this would require [n_fef, n_ref, 2] and [n_test, n_test, 2] idx tensors

        # Initialise regularisation parameters
        # Implemented only for first _cmmd call which corresponds to original window assignment
        if self.lams is None:
            possible_lams = tf.convert_to_tensor([2**(-i) for i in range(20)])  # fairly arbitrary atm # TODO
            lam_0 = self._pick_lam(possible_lams, K_0, L_0, n_folds=5)
            lam_1 = self._pick_lam(possible_lams, K_1, L_1, n_folds=5)
            self.lams = (lam_0, lam_1)  # type: ignore[assignment]
            # Ignore above as self.lams is Optional[Tuple[float, float]] in base class.

        # Compute stat
        L_0_inv = tf.linalg.inv(L_0 + n_ref*self.lams[0]*tf.eye(int(n_ref)))
        L_1_inv = tf.linalg.inv(L_1 + n_test*self.lams[1]*tf.eye(int(n_test)))
        A_0 = tf.gather(L_held, idx_0, axis=1) @ L_0_inv
        A_1 = tf.gather(L_held, idx_1, axis=1) @ L_1_inv
        # Allow batches of MMDs to be computed at a time (rather than all)
        if self.batch_size is not None:
            bs = self.batch_size
            coupling_xx = tf.reduce_mean(tf.stack([tf.reduce_mean(tf.einsum('ij,ik->ijk', A_0_i, A_0_i), axis=0)
                                                   for A_0_i in tf.split(A_0, _split_chunks(len(A_0), bs))]), axis=0)
            coupling_yy = tf.reduce_mean(tf.stack([tf.reduce_mean(tf.einsum('ij,ik->ijk', A_1_i, A_1_i), axis=0)
                                                   for A_1_i in tf.split(A_1, _split_chunks(len(A_1), bs))]), axis=0)
            coupling_xy = tf.reduce_mean(tf.stack([
                tf.reduce_mean(tf.einsum('ij,ik->ijk', A_0_i, A_1_i), axis=0)
                for A_0_i, A_1_i in zip(tf.split(A_0, _split_chunks(len(A_0), bs)),
                                        tf.split(A_1, _split_chunks(len(A_1), bs)))
            ]), axis=0)
        else:
            coupling_xx = tf.reduce_mean(tf.einsum('ij,ik->ijk', A_0, A_0), axis=0)
            coupling_yy = tf.reduce_mean(tf.einsum('ij,ik->ijk', A_1, A_1), axis=0)
            coupling_xy = tf.reduce_mean(tf.einsum('ij,ik->ijk', A_0, A_1), axis=0)
        sim_xx = tf.reduce_sum(tf.gather(tf.gather(K, idx_0), idx_0, axis=1)*coupling_xx)
        sim_yy = tf.reduce_sum(tf.gather(tf.gather(K, idx_1), idx_1, axis=1)*coupling_yy)
        sim_xy = tf.reduce_sum(tf.gather(tf.gather(K, idx_0), idx_1, axis=1)*coupling_xy)
        stat = sim_xx + sim_yy - 2*sim_xy

        return stat, coupling_xx, coupling_yy, coupling_xy

    def _pick_lam(self, lams: tf.Tensor, K: tf.Tensor, L: tf.Tensor, n_folds: int = 5) -> tf.Tensor:
        """
        The conditional mean embedding is estimated as the solution of a regularised regression problem.

        This private method function uses cross validation to select the regularisation parameter that
        minimises squared error on the out-of-fold instances. The error is a distance in the RKHS and is
        therefore an MMD-like quantity itself.
        """
        n = len(L)
        fold_size = n // n_folds
        losses = tf.zeros_like(lams, dtype=tf.float32)
        for fold in range(n_folds):
            inds_oof = np.arange(n)[(fold*fold_size):((fold+1)*fold_size)]
            inds_if = np.setdiff1d(np.arange(n), inds_oof)
            K_if = tf.gather(tf.gather(K, inds_if), inds_if, axis=1)
            L_if = tf.gather(tf.gather(L, inds_if), inds_if, axis=1)
            n_if = len(K_if)
            L_inv_lams = tf.stack(
                [tf.linalg.inv(L_if + n_if*lam*tf.eye(n_if)) for lam in lams])  # n_lam x n_if x n_if
            KW = tf.einsum('ij,ljk->lik', K_if, L_inv_lams)
            lW = tf.einsum('ij,ljk->lik', tf.gather(tf.gather(L, inds_oof), inds_if, axis=1), L_inv_lams)
            lWKW = tf.einsum('lij,ljk->lik', lW, KW)
            lWKWl = tf.einsum('lkj,jk->lk', lWKW, tf.gather(tf.gather(L, inds_if), inds_oof, axis=1))  # n_lam x n_oof
            lWk = tf.einsum('lij,ji->li', lW, tf.gather(tf.gather(K, inds_if), inds_oof, axis=1))  # n_lam x n_oof
            kxx = tf.ones_like(lWk) * tf.reduce_max(K)
            losses += tf.reduce_sum(lWKWl + kxx - 2*lWk, axis=-1)
        return lams[tf.argmin(losses)]


def _split_chunks(n: int, p: int) -> List[int]:
    """
    Private function to calculate chunk sizes for tf.split(). An array/tensor of length n is aimed to be split into p
    number of chunks of roughly equivalent size.

    Parameters
    ----------
    n
        Size of array/tensor to be split.
    p
        Number of chunks.

    Returns
    -------
    List containing the size of each chunk.
    """
    if p >= n:
        chunks = [n]
    else:
        chunks = [n // p + 1] * (n % p) + [n // p] * (p - n % p)
    return chunks
