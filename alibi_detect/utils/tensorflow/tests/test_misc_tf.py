import tensorflow as tf
from alibi_detect.utils.tensorflow import zero_diag


def test_zero_diag():
    ones = tf.ones((10, 10))
    ones_zd = zero_diag(ones)
    assert ones_zd.shape == (10, 10)
    assert float(tf.linalg.trace(ones_zd)) == 0
    assert float(tf.reduce_sum(ones_zd)) == 90
