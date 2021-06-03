import logging
import torch
from typing import Callable, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)


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


def permed_lsdds(
    k_all_c: torch.Tensor,
    x_perms: List[torch.Tensor],
    y_perms: List[torch.Tensor],
    H: torch.Tensor,
    H_lam_inv: Optional[torch.Tensor] = None,
    lam_rd_max: float = 0.2,
    return_unpermed: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute LSDD estimates from kernel matrix across various ref and test window samples

    Parameters
    ----------
    k_all_c
        Kernel matrix of simmilarities between all samples and the kernel centers.
    x_perms
        List of B reference window index vectors
    y_perms
        List of B test window index vectors
    H
        Special (scaled) kernel matrix of simmilarities between kernel centers
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
    k_xc_perms = torch.stack([k_all_c[x_inds] for x_inds in x_perms], 0)
    k_yc_perms = torch.stack([k_all_c[y_inds] for y_inds in y_perms], 0)
    h_perms = k_xc_perms.mean(1) - k_yc_perms.mean(1)

    if H_lam_inv is None:
        # We perform the initialisation for multiple candidate lambda values and pick the largest
        # one for which the relative difference (RD) between two difference estimates is below lambda_rd_max.
        # See Appendix A
        candidate_lambdas = [1/(4**i) for i in range(10)]  # TODO: More principled selection
        H_plus_lams = torch.stack(
            [H+torch.eye(H.shape[0], device=H.device)*can_lam for can_lam in candidate_lambdas], 0
        )
        H_plus_lam_invs = torch.inverse(H_plus_lams)
        H_plus_lam_invs = H_plus_lam_invs.permute(1, 2, 0)  # put lambdas in final axis

        omegas = torch.einsum('jkl,bk->bjl', H_plus_lam_invs, h_perms)  # (Eqn 8)
        h_omegas = torch.einsum('bj,bjl->bl', h_perms, omegas)
        omega_H_omegas = torch.einsum('bkl,bkl->bl', torch.einsum('bjl,jk->bkl', omegas, H), omegas)
        rds = (1 - (omega_H_omegas/h_omegas)).mean(0)
        lam_index = (rds < lam_rd_max).nonzero()[0]
        lam = candidate_lambdas[lam_index]
        logger.info(f"Using lambda value of {lam:.2g} with RD of {float(rds[lam_index]):.2g}")
        H_plus_lam_inv = H_plus_lam_invs[:, :, lam_index.item()]
        H_lam_inv = 2*H_plus_lam_inv - (H_plus_lam_inv.transpose(0, 1) @ H @ H_plus_lam_inv)  # (below Eqn 11)

    # Now to compute an LSDD estimate for each permutation
    lsdd_perms = (h_perms * (H_lam_inv @ h_perms.transpose(0, 1)).transpose(0, 1)).sum(-1)  # (Eqn 11)

    if return_unpermed:
        n_x = x_perms[0].shape[0]
        h = k_all_c[:n_x].mean(0) - k_all_c[n_x:].mean(0)
        lsdd_unpermed = (h[None, :] * (H_lam_inv @ h[:, None]).transpose(0, 1)).sum()
        return lsdd_perms, H_lam_inv, lsdd_unpermed
    else:
        return lsdd_perms, H_lam_inv
