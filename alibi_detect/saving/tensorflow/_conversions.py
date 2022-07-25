import tensorflow as tf


def get_tf_dtype(dtype_str: str):
    """Returns tensorflow datatype specified by string."""

    return getattr(tf, dtype_str)
