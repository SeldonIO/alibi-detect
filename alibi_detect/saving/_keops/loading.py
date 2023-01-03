from typing import Callable
from alibi_detect.utils.keops.kernels import DeepKernel


def load_kernel_config(cfg: dict) -> Callable:
    """
    Loads a kernel from a kernel config dict.

    Parameters
    ----------
    cfg
        A kernel config dict. (see pydantic schema's).

    Returns
    -------
    The kernel.
    """
    if 'src' in cfg:  # Standard kernel config
        kernel = cfg.pop('src')
        if hasattr(kernel, 'from_config'):
            kernel = kernel.from_config(cfg)

    elif 'proj' in cfg:  # DeepKernel config
        # Kernel a
        kernel_a = cfg['kernel_a']
        kernel_b = cfg['kernel_b']
        if kernel_a != 'rbf':
            cfg['kernel_a'] = load_kernel_config(kernel_a)
        if kernel_b != 'rbf':
            cfg['kernel_b'] = load_kernel_config(kernel_b)
        # Assemble deep kernel
        kernel = DeepKernel.from_config(cfg)

    else:
        raise ValueError('Unable to process kernel. The kernel config dict must either be a `KernelConfig` with a '
                         '`src` field, or a `DeepkernelConfig` with a `proj` field.)')
    return kernel
