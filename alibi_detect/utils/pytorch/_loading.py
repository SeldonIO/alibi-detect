# from pathlib import Path
import torch
# import os
from typing import Optional, Callable
# from alibi_detect.utils._types import Literal
# from alibi_detect.models.pytorch import TransformerEmbedding
from alibi_detect.utils.pytorch.kernels import GaussianRBF, DeepKernel
import numpy as np
import logging

logger = logging.getLogger(__name__)


# def load_model():  TODO
#    """
#    """


# def _prep_model_and_emb() -> Callable:
#    """
#    """


def load_kernel_config(cfg: dict, device: Optional[str] = None) -> Callable:
    """
    Loads a kernel from a kernel config dict.

    Parameters
    ----------
    cfg
        A kernel config dict. (see pydantic schema's).
    device
        Device type used. The default None tries to use the GPU and falls back on CPU if needed.

    Returns
    -------
    The kernel.
    """
    if 'src' in cfg:  # Standard kernel config
        kernel = cfg['src']
        sigma = cfg['sigma']
        if callable(kernel):
            if kernel.__name__ == 'GaussianRBF':
                torch_device = set_device(device)
                sigma = torch.from_numpy(sigma).to(torch_device) if isinstance(sigma, np.ndarray) else None
                kernel = kernel(sigma=sigma, trainable=cfg['trainable'])
            else:
                kwargs = cfg['kwargs']
                kernel = kernel(**kwargs)

    elif 'proj' in cfg:  # DeepKernel config
        proj = cfg['proj']
        eps = cfg['eps']
        # Kernel a
        kernel_a = cfg['kernel_a']
        if kernel_a is not None:
            kernel_a = load_kernel_config(kernel_a)
        else:
            kernel_a = GaussianRBF(trainable=True)
        # Kernel b
        kernel_b = cfg['kernel_b']
        if kernel_b is not None:
            kernel_b = load_kernel_config(kernel_b)
        else:
            kernel_b = GaussianRBF(trainable=True)

        # Assemble deep kernel
        kernel = DeepKernel(proj, kernel_a=kernel_a, kernel_b=kernel_b, eps=eps)
    else:
        raise ValueError('Unable to process kernel.)')
    return kernel


def set_device(device: Optional[str]) -> torch.device:
    """
    Set PyTorch device.

    Parameters
    ----------
    device
        String identifying the device.

    Returns
    -------
    A set torch.device object.
    """
    if device is None or device in ['gpu', 'cuda']:
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch_device.type == 'cpu':
            logger.warning('No GPU detected, fall back on CPU.')
    else:
        torch_device = torch.device('cpu')
    return torch_device
