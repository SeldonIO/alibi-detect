import tensorflow as tf


def zero_diag(mat: tf.Tensor) -> tf.Tensor:
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
    return mat - tf.linalg.diag(tf.linalg.diag_part(mat))
