import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


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


def quantile(sample: torch.Tensor, p: float, type: int = 7, sorted: bool = False) -> float:
    """
    Estimate a desired quantile of a univariate distribution from a vector of samples

    Parameters
    ----------
    sample
        A 1D vector of values
    p
        The desired quantile in (0,1)
    type
        The method for computing the quantile.
        See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
    sorted
        Whether or not the vector is already sorted into ascending order

    Returns
    -------
    An estimate of the quantile

    """
    N = len(sample)

    if len(sample.shape) != 1:
        raise ValueError("Quantile estimation only supports vectors of univariate samples.")
    if not 1/N <= p <= (N-1)/N:
        raise ValueError(f"The {p}-quantile should not be estimated using only {N} samples.")

    sorted_sample = sample if sorted else sample.sort().values

    if type == 6:
        h = (N+1)*p
    elif type == 7:
        h = (N-1)*p + 1
    elif type == 8:
        h = (N+1/3)*p + 1/3
    h_floor = int(h)
    quantile = sorted_sample[h_floor-1]
    if h_floor != h:
        quantile += (h - h_floor)*(sorted_sample[h_floor]-sorted_sample[h_floor-1])

    return float(quantile)


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Instantiates a PyTorch device object.

    Parameters
    ----------
    device
        Either `None`, a str ('gpu' or 'cpu') indicating the device to choose, or an already instantiated device
        object. If `None`, the GPU is selected if it is detected, otherwise the CPU is used as a fallback.

    Returns
    -------
    The instantiated device object.
    """
    if isinstance(device, torch.device):  # Already a torch device
        return device
    else:  # Instantiate device
        if device is None or device.lower() in ['gpu', 'cuda']:
            torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch_device.type == 'cpu':
                logger.warning('No GPU detected, fall back on CPU.')
        else:
            torch_device = torch.device('cpu')
            if device.lower() != 'cpu':
                logger.warning('Requested device not recognised, fall back on CPU.')
    return torch_device
