import logging
import numpy as np
import tensorflow as tf
from typing import Callable, Tuple, List, Optional, Union

logger = logging.getLogger(__name__)


def squared_pairwise_distance(x: tf.Tensor, y: tf.Tensor, a_min: float = 1e-30, a_max: float = 1e30) -> tf.Tensor:
    """
    TensorFlow pairwise squared Euclidean distance between samples x and y.

    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    a_min
        Lower bound to clip distance values.
    a_max
        Upper bound to clip distance values.

    Returns
    -------
    Pairwise squared Euclidean distance [Nx, Ny].
    """
    x2 = tf.reduce_sum(x ** 2, axis=-1, keepdims=True)
    y2 = tf.reduce_sum(y ** 2, axis=-1, keepdims=True)
    dist = x2 + tf.transpose(y2, (1, 0)) - 2. * x @ tf.transpose(y, (1, 0))
    return tf.clip_by_value(dist, a_min, a_max)


def batch_compute_kernel_matrix(
    x: Union[list, np.ndarray, tf.Tensor],
    y: Union[list, np.ndarray, tf.Tensor],
    kernel: Union[Callable, tf.keras.Model],
    batch_size: int = int(1e10),
    preprocess_fn: Callable = None,
) -> tf.Tensor:
    """
    Compute the kernel matrix between x and y by filling in blocks of size
    batch_size x batch_size at a time.

    Parameters
    ----------
    x
        Reference set.
    y
        Test set.
    kernel
        tf.keras model
    batch_size
        Batch size used during prediction.
    preprocess_fn
        Optional preprocessing function for each batch.

    Returns
    -------
    Kernel matrix in the form of a tensorflow tensor
    """
    if type(x) != type(y):
        raise ValueError("x and y should be of the same type")

    n_x, n_y = len(x), len(y)
    n_batch_x, n_batch_y = int(np.ceil(n_x / batch_size)), int(np.ceil(n_y / batch_size))

    k_is = []
    for i in range(n_batch_x):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n_x)
        x_batch = x[istart:istop]
        if isinstance(preprocess_fn, Callable):  # type: ignore
            x_batch = preprocess_fn(x_batch)
        k_ijs = []
        for j in range(n_batch_y):
            jstart, jstop = j * batch_size, min((j + 1) * batch_size, n_y)
            y_batch = y[jstart:jstop]
            if isinstance(preprocess_fn, Callable):  # type: ignore
                y_batch = preprocess_fn(y_batch)
            k_ijs.append(kernel(x_batch, y_batch))  # type: ignore
        k_is.append(tf.concat(k_ijs, axis=1))
    k_mat = tf.concat(k_is, axis=0)
    return k_mat


def mmd2_from_kernel_matrix(kernel_mat: tf.Tensor, m: int, permute: bool = False,
                            zero_diag: bool = True) -> tf.Tensor:
    """
    Compute maximum mean discrepancy (MMD^2) between 2 samples x and y from the
    full kernel matrix between the samples.

    Parameters
    ----------
    kernel_mat
        Kernel matrix between samples x and y.
    m
        Number of instances in y.
    permute
        Whether to permute the row indices. Used for permutation tests.
    zero_diag
        Whether to zero out the diagonal of the kernel matrix.

    Returns
    -------
    MMD^2 between the samples from the kernel matrix.
    """
    n = kernel_mat.shape[0] - m
    if zero_diag:
        kernel_mat = kernel_mat - tf.linalg.diag(tf.linalg.diag_part(kernel_mat))
    if permute:
        idx = np.random.permutation(kernel_mat.shape[0])
        kernel_mat = tf.gather(tf.gather(kernel_mat, indices=idx, axis=0), indices=idx, axis=1)
    k_xx, k_yy, k_xy = kernel_mat[:-m, :-m], kernel_mat[-m:, -m:], kernel_mat[-m:, :-m]
    c_xx, c_yy = 1 / (n * (n - 1)), 1 / (m * (m - 1))
    mmd2 = c_xx * tf.reduce_sum(k_xx) + c_yy * tf.reduce_sum(k_yy) - 2. * tf.reduce_mean(k_xy)
    return mmd2


def mmd2(x: tf.Tensor, y: tf.Tensor, kernel: Callable) -> float:
    """
    Compute MMD^2 between 2 samples.

    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    kernel
        Kernel function.

    Returns
    -------
    MMD^2 between the samples x and y.
    """
    n, m = x.shape[0], y.shape[0]
    c_xx, c_yy = 1 / (n * (n - 1)), 1 / (m * (m - 1))
    k_xx, k_yy, k_xy = kernel(x, x), kernel(y, y), kernel(x, y)  # type: ignore
    return (c_xx * (tf.reduce_sum(k_xx) - tf.linalg.trace(k_xx)) +
            c_yy * (tf.reduce_sum(k_yy) - tf.linalg.trace(k_yy)) - 2. * tf.reduce_mean(k_xy))


def relative_euclidean_distance(x: tf.Tensor, y: tf.Tensor, eps: float = 1e-12, axis: int = -1) -> tf.Tensor:
    """
    Relative Euclidean distance.

    Parameters
    ----------
    x
        Tensor used in distance computation.
    y
        Tensor used in distance computation.
    eps
        Epsilon added to denominator for numerical stability.
    axis
        Axis used to compute distance.

    Returns
    -------
    Tensor with relative Euclidean distance across specified axis.
    """
    denom = tf.concat([tf.reshape(tf.norm(x, ord=2, axis=axis), (-1, 1)),
                       tf.reshape(tf.norm(y, ord=2, axis=axis), (-1, 1))], axis=1)
    dist = tf.norm(x - y, ord=2, axis=axis) / (tf.reduce_min(denom, axis=axis) + eps)
    return dist


def permed_lsdds(
    k_all_c: tf.Tensor,
    x_perms: List[tf.Tensor],
    y_perms: List[tf.Tensor],
    H: tf.Tensor,
    H_lam_inv: Optional[tf.Tensor] = None,
    lam_rd_max: float = 0.2,
    return_unpermed: bool = False,
) -> Union[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """
    Compute LSDD estimates from kernel matrix across various ref and test window samples

    Parameters
    ----------
    k_all_c
        Kernel matrix of similarities between all samples and the kernel centers.
    x_perms
        List of B reference window index vectors
    y_perms
        List of B test window index vectors
    H
        Special (scaled) kernel matrix of similarities between kernel centers
    H_lam_inv
        Function of H corresponding to a particular regulariation parameter lambda.
        See Eqn 11 of Bu et al. (2017)
    lam_rd_max
        The maximum relative difference between two estimates of LSDD that the regularization parameter
        lambda is allowed to cause. Defaults to 0.2. Only relavent if H_lam_inv is not supplied.
    return_unpermed
        Whether or not to return value corresponding to unpermed order defined by k_all_c

    Returns
    -------
    Vector of B LSDD estimates for each permutation, H_lam_inv which may have been inferred, and optionally
    the unpermed LSDD estimate.
    """

    # Compute (for each bootstrap) the average distance to each kernel center (Eqn 7)
    k_xc_perms = tf.stack([tf.gather(k_all_c, x_inds) for x_inds in x_perms], axis=0)
    k_yc_perms = tf.stack([tf.gather(k_all_c, y_inds) for y_inds in y_perms], axis=0)
    h_perms = tf.reduce_mean(k_xc_perms, axis=1) - tf.reduce_mean(k_yc_perms, axis=1)

    if H_lam_inv is None:
        # We perform the initialisation for multiple candidate lambda values and pick the largest
        # one for which the relative difference (RD) between two difference estimates is below lambda_rd_max.
        # See Appendix A
        candidate_lambdas = [1/(4**i) for i in range(10)]  # TODO: More principled selection
        H_plus_lams = tf.stack([H+tf.eye(H.shape[0], dtype=H.dtype)*can_lam for can_lam in candidate_lambdas], axis=0)
        H_plus_lam_invs = tf.transpose(tf.linalg.inv(H_plus_lams), [1, 2, 0])  # lambdas last
        omegas = tf.einsum('jkl,bk->bjl', H_plus_lam_invs, h_perms)  # (Eqn 8)
        h_omegas = tf.einsum('bj,bjl->bl', h_perms, omegas)
        omega_H_omegas = tf.einsum('bkl,bkl->bl', tf.einsum('bjl,jk->bkl', omegas, H), omegas)
        rds = tf.reduce_mean(1 - (omega_H_omegas/h_omegas), axis=0)
        less_than_rd_inds = tf.where(rds < lam_rd_max)
        if len(less_than_rd_inds) == 0:
            repeats = k_all_c.shape[0] - np.unique(k_all_c, axis=0).shape[0]
            if repeats > 0:
                msg = "Too many repeat instances for LSDD-based detection. \
                Try using MMD-based detection instead"
            else:
                msg = "Unknown error. Try using MMD-based detection instead"
            raise ValueError(msg)
        lambda_index = int(less_than_rd_inds[0])
        lam = candidate_lambdas[lambda_index]
        logger.info(f"Using lambda value of {lam:.2g} with RD of {float(rds[lambda_index]):.2g}")
        H_plus_lam_inv = tf.linalg.inv(H+lam*tf.eye(H.shape[0], dtype=H.dtype))
        H_lam_inv = 2*H_plus_lam_inv - (tf.transpose(H_plus_lam_inv, [1, 0]) @ H @ H_plus_lam_inv)  # (blw Eqn 11)

    # Now to compute an LSDD estimate for each permutation
    lsdd_perms = tf.reduce_sum(
        h_perms * tf.transpose(H_lam_inv @ tf.transpose(h_perms, [1, 0]), [1, 0]), axis=1
    )  # (Eqn 11)

    if return_unpermed:
        n_x = x_perms[0].shape[0]
        h = tf.reduce_mean(k_all_c[:n_x], axis=0) - tf.reduce_mean(k_all_c[n_x:], axis=0)
        lsdd_unpermed = tf.reduce_sum(h[None, :] * tf.transpose(H_lam_inv @ h[:, None], [1, 0]))
        return lsdd_perms, H_lam_inv, lsdd_unpermed
    else:
        return lsdd_perms, H_lam_inv
