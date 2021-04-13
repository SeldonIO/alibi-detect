import torch
from typing import Callable


@torch.jit.script
def squared_pairwise_distance(x: torch.Tensor, y: torch.Tensor, a_min: float = 1e-30) -> torch.Tensor:
    """
    PyTorch pairwise squared Euclidean distance between samples x and y.

    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    a_min
        Lower bound to clip distance values.
    Returns
    -------
    Pairwise squared Euclidean distance [Nx, Ny].
    """
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    dist = torch.addmm(y2.transpose(-2, -1), x, y.transpose(-2, -1), alpha=-2).add_(x2)
    return dist.clamp_min_(a_min)


def mmd2_from_kernel_matrix(kernel_mat: torch.Tensor, m: int, permute: bool = False,
                            zero_diag: bool = True) -> torch.Tensor:
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
        kernel_mat = kernel_mat - torch.diag(kernel_mat.diag())
    if permute:
        idx = torch.randperm(kernel_mat.shape[0])
        kernel_mat = kernel_mat[idx][:, idx]
    k_xx, k_yy, k_xy = kernel_mat[:-m, :-m], kernel_mat[-m:, -m:], kernel_mat[-m:, :-m]
    c_xx, c_yy = 1 / (n * (n - 1)), 1 / (m * (m - 1))
    mmd2 = c_xx * k_xx.sum() + c_yy * k_yy.sum() - 2. * k_xy.mean()
    return mmd2


def mmd2(x: torch.Tensor, y: torch.Tensor, kernel: Callable) -> float:
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
    return c_xx * (k_xx.sum() - k_xx.trace()) + c_yy * (k_yy.sum() - k_yy.trace()) - 2. * k_xy.mean()
