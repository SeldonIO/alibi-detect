import dask.array as da
import numpy as np
from typing import Union
from . import distance


def gaussian_kernel(x: Union[np.ndarray, da.array],
                    y: Union[np.ndarray, da.array],
                    sigma: np.ndarray = None
                    ) -> Union[np.ndarray, da.array]:
    """
    Gaussian kernel between samples of x and y. A sum of kernels is computed
    for multiple values of sigma.

    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    sigma
        Array with values of the kernel width sigma.
    chunks
        Chunk sizes for x and y when using dask to compute the pairwise distances.

    Returns
    -------
    Array [Nx, Ny] of the kernel.
    """
    beta = 1. / (2. * np.expand_dims(sigma, 1))  # [Ns,1]
    dist = distance.pairwise_distance(x, y)  # [Nx,Ny]
    s = beta @ dist.reshape(1, -1)  # [Ns,1]*[1,Nx*Ny]=[Ns,Nx*Ny]
    s = np.exp(-s) if isinstance(s, np.ndarray) else da.exp(-s)
    return s.sum(axis=0).reshape(dist.shape)  # [Nx,Ny]


def infer_sigma(x: Union[np.ndarray, da.array],
                y: Union[np.ndarray, da.array],
                p: int = 2
                ) -> float:
    """
    Infer sigma used in the kernel by setting it to the median distance
    between each of the pairwise instances in x and y.

    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    p
        Power used in the distance calculation, default equals 2 (Euclidean distance).
    chunks
        Chunk sizes for x and y when using dask to compute the pairwise distances.

    Returns
    -------
    Sigma used in the kernel.
    """
    dist = distance.pairwise_distance(x, y, p=p)
    return np.median(dist) if isinstance(dist, np.ndarray) else da.median(dist.reshape(-1,), axis=0).compute()
