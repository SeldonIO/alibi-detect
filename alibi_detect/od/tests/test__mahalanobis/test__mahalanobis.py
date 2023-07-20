import pytest
import numpy as np
import torch

from alibi_detect.od._mahalanobis import Mahalanobis
from alibi_detect.exceptions import NotFittedError
from sklearn.datasets import make_moons


def make_mahalanobis_detector():
    mahalanobis_detector = Mahalanobis()
    x_ref = np.random.randn(100, 2)
    mahalanobis_detector.fit(x_ref)
    mahalanobis_detector.infer_threshold(x_ref, 0.1)
    return mahalanobis_detector


def test_unfitted_mahalanobis_single_score():
    """Test Mahalanobis detector throws errors when not fitted."""
    mahalanobis_detector = Mahalanobis()
    x = np.array([[0, 10], [0.1, 0]])
    x_ref = np.random.randn(100, 2)

    with pytest.raises(NotFittedError) as err:
        mahalanobis_detector.infer_threshold(x_ref, 0.1)
    assert str(err.value) == 'Mahalanobis has not been fit!'

    with pytest.raises(NotFittedError) as err:
        mahalanobis_detector.score(x)
    assert str(err.value) == 'Mahalanobis has not been fit!'

    # test predict raises exception when not fitted
    with pytest.raises(NotFittedError) as err:
        mahalanobis_detector.predict(x)
    assert str(err.value) == 'Mahalanobis has not been fit!'


def test_fitted_mahalanobis_score():
    """Test Mahalanobis detector score method.

    Test Mahalanobis detector that has been fitted on reference data but has not had a threshold
    inferred can still score data using the predict method. Test that it does not raise an error
    but does not return `threshold`, `p_value` and `is_outlier` values.
    """
    mahalanobis_detector = Mahalanobis()
    x_ref = np.random.randn(100, 2)
    mahalanobis_detector.fit(x_ref)
    x = np.array([[0, 10], [0.1, 0]])
    scores = mahalanobis_detector.score(x)

    y = mahalanobis_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 5
    assert y['instance_score'][1] < 1
    assert all(y['instance_score'] == scores)
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


def test_fitted_mahalanobis_predict():
    """Test Mahalanobis detector predict method.

    Test Mahalanobis detector that has been fitted on reference data and has had a threshold
    inferred can score data using the predict method as well as predict outliers. Test that it
    returns `threshold`, `p_value` and `is_outlier` values.
    """
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
    assert isinstance(y['threshold'], float)
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


def test_mahalanobis_integration(tmp_path):
    """Test Mahalanobis detector on moons dataset.

    Test Mahalanobis detector on a more complex 2d example. Test that the detector can be fitted
    on reference data and infer a threshold. Test that it differentiates between inliers and outliers.
    Test that the detector can be scripted.
    """
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

    ts_mahalanobis.save(tmp_path / 'mahalanobis.pt')
    mahalanobis_detector = torch.load(tmp_path / 'mahalanobis.pt')
    y = mahalanobis_detector(x)
    assert torch.all(y == torch.tensor([False, True]))
