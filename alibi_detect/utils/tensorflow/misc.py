import tensorflow as tf
import warnings


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


def quantile(sample: tf.Tensor, p: float, type: int = 7, sorted: bool = False) -> float:
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

    sorted_sample = sample if sorted else tf.sort(sample)

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


def subset_matrix(mat: tf.Tensor, inds_0: tf.Tensor, inds_1: tf.Tensor) -> tf.Tensor:
    """
    Take a matrix and return the submatrix correspond to provided row and column indices

    Parameters
    ----------
    mat
        A 2D matrix
    inds_0
        A vector of row indices
    inds_1
        A vector of column indices

    Returns
    -------
    A submatrix of shape (len(inds_0), len(inds_1))
    """
    if len(mat.shape) != 2:
        raise ValueError("Subsetting only supported for matrices (2D)")
    subbed_rows = tf.gather(mat, inds_0, axis=0)
    subbed_rows_cols = tf.gather(subbed_rows, inds_1, axis=1)
    return subbed_rows_cols


def clone_model(model: tf.keras.Model) -> tf.keras.Model:
    """ Clone a sequential, functional or subclassed tf.keras.Model. """
    try:  # sequential or functional model
        return tf.keras.models.clone_model(model)
    except ValueError:  # subclassed model
        try:
            config = model.get_config()
        except NotImplementedError:
            config = {}
        return model.__class__.from_config(config)


def check_model(model: tf.keras.Model, raise_error: bool = False) -> None:
    """

    Parameters
    ----------
    model
    raise_error

    Raises
    ------

    """
    # Message to display if there are problems
    msg = "The TensorFlow model may have been loaded incorrectly. This could be because `get_config` and/or " \
          "`from_config` methods were defined incorrectly. Otherwise, it could be because custom objects have not " \
          "been provided. For more guidance see the TensorFlow tab at " \
          "https://docs.seldon.io/projects/alibi-detect/en/stable/overview/saving.html#supported-ml-models."

    try:
        # Check if model is a `RevivedNetwork` rather than the real original model (this occurs when subclassed models
        # are loaded without all custom objects being provided
        # Note, could also do `if model.__class__.__base__.__name__ == 'RevivedNetwork':`
        if model.__class__.__module__ == 'keras.saving.saved_model.load':
            # Raise error (this will be caught and re-raised below)
            raise ValueError('The model appears to be a `keras.saving.saved_model.load.RevivedNetwork. This suggests '
                             'a subclassed model has been loaded without all custom objects being provided.')

        # Check model cloning doesn't raise error
        clone_model(model)

    except Exception as error:
        if raise_error:
            raise ValueError(msg) from error
        else:
            warnings.warn(msg + f"Original error message: \n\t{error}")
