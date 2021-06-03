import torch


def zero_diag(mat: torch.Tensor) -> torch.Tensor:
    """
    Set the diagonal of a matrix to 0

    Parameters
    ----------
    mat
        A 2D square matrix

    Returns
    -------
    A 2D square matrix with zeros along the diagonal
    """
    return mat - torch.diag(mat.diag())
