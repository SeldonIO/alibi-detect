import numpy as np\

from alibi_detect.od.transforms import PValNormaliser, ShiftAndScaleNormaliser, TopKAggregator, MaxAggregator, \
    MinAggregator, AverageAggregator
import pytest


def test_p_val_normaliser():
    p_val_norm = PValNormaliser()
    scores = np.random.normal(0, 1, (1000, 3)) * np.array([0.5, 2, 1]) + np.array([0, 1, -1])

    with pytest.raises(ValueError):
        p_val_norm.transform(np.array([[0, 0, 0]]))

    p_val_norm.fit(X=scores)

    s = np.array([[0, 0, 0]]) + np.array([0, 1, -1])
    s_transformed = p_val_norm.transform(s)
    np.testing.assert_array_almost_equal(
        np.array([[0.5, 0.5, 0.5]]),
        s_transformed,
        decimal=0.1
    )

    s = np.array([[0, 0, 0]])
    s_transformed = p_val_norm.transform(s)
    np.testing.assert_array_almost_equal(
        np.array([[0.5, 0.32, 0.85]]),
        s_transformed,
        decimal=0.1
    )


def test_shift_and_scale_normaliser():
    shift_and_scale_norm = ShiftAndScaleNormaliser()
    scores = np.random.normal(0, 1, (1000, 3)) * np.array([0.5, 2, 1]) + np.array([0, 1, -1])
    shift_and_scale_norm.fit(X=scores)
    s = np.array([[0, 0, 0]]) + np.array([0, 1, -1])
    s_transformed = shift_and_scale_norm.transform(s)
    np.testing.assert_array_almost_equal(
        np.array([[0., 0., 0.]]),
        s_transformed,
        decimal=0.1
    )

    s = np.array([[0, 0, 0]])
    s_transformed = shift_and_scale_norm.transform(s)
    np.testing.assert_array_almost_equal(
        np.array([[0., -0.5, 1.]]),
        s_transformed,
        decimal=0.1
    )
