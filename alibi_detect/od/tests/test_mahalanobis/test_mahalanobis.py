import pytest
import numpy as np
import torch

from alibi_detect.od.mahalanobis import Mahalanobis
from alibi_detect.od.base import NotFitException
from sklearn.datasets import make_moons


def make_mahalanobis_detector():
    mahalanobis_detector = Mahalanobis()
    x_ref = np.random.randn(100, 2)
    mahalanobis_detector.fit(x_ref)
    mahalanobis_detector.infer_threshold(x_ref, 0.1)
    return mahalanobis_detector


def test_unfitted_mahalanobis_single_score():
    mahalanobis_detector = Mahalanobis()
    x = np.array([[0, 10], [0.1, 0]])
    with pytest.raises(NotFitException) as err:
        _ = mahalanobis_detector.predict(x)
    assert str(err.value) == 'MahalanobisTorch has not been fit!'


def test_fitted_mahalanobis_single_score():
    mahalanobis_detector = Mahalanobis()
    x_ref = np.random.randn(100, 2)
    mahalanobis_detector.fit(x_ref)
    x = np.array([[0, 10], [0.1, 0]])
    y = mahalanobis_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 5
    assert y['instance_score'][1] < 1
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


def test_fitted_mahalanobis_predict():
    mahalanobis_detector = make_mahalanobis_detector()
    x_ref = np.random.randn(100, 2)
    mahalanobis_detector.infer_threshold(x_ref, 0.1)
    x = np.array([[0, 10], [0, 0.1]])
    y = mahalanobis_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 5
    assert y['instance_score'][1] < 1
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


def test_mahalanobis_integration():
    mahalanobis_detector = Mahalanobis()
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    mahalanobis_detector.fit(X_ref)
    mahalanobis_detector.infer_threshold(X_ref, 0.1)
    result = mahalanobis_detector.predict(x_inlier)
    result = result['data']['is_outlier'][0]
    assert not result

    x_outlier = np.array([[-1, 1.5]])
    result = mahalanobis_detector.predict(x_outlier)
    result = result['data']['is_outlier'][0]
    assert result

    ts_mahalanobis = torch.jit.script(mahalanobis_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = ts_mahalanobis(x)
    assert torch.all(y == torch.tensor([False, True]))
