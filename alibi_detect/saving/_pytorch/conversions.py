import torch


def get_pt_dtype(dtype_str: str):
    """Returns pytorch datatype specified by string."""

    return getattr(torch, dtype_str)
