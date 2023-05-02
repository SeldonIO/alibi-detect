import pytest
import numpy as np
import torch

from alibi_detect.od._gmm import GMM
from alibi_detect.exceptions import NotFittedError

from sklearn.datasets import make_moons


@pytest.mark.parametrize('backend', ['pytorch', 'sklearn'])
def test_unfitted_gmm_single_score(backend):
    """
    test predict raises exception when not fitted
    """
    gmm_detector = GMM(n_components=1, backend=backend)
    x = np.array([[0, 10], [0.1, 0]])

    with pytest.raises(NotFittedError) as err:
        _ = gmm_detector.predict(x)
    assert str(err.value) == f'{gmm_detector.__class__.__name__} has not been fit!'


@pytest.mark.parametrize('backend', ['pytorch', 'sklearn'])
def test_fitted_gmm_single_score(backend):
    """
    Test that a detector that has been fitted on data but that has not got an inferred
    threshold, will correctly score outliers using the predict method.
    """
    gmm_detector = GMM(n_components=1, backend=backend)
    x_ref = np.random.randn(100, 2)
    gmm_detector.fit(x_ref)
    x = np.array([[0, 10], [0.1, 0]])
    y = gmm_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 5
    assert y['instance_score'][1] < 2
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


@pytest.mark.parametrize('backend', ['pytorch', 'sklearn'])
def test_fitted_gmm_predict(backend):
    """
    Test that a detector that has been fitted on data and with an inferred threshold,
    will correctly score and label outliers, as well as return the p-values using the
    predict method.
    """
    gmm_detector = GMM(n_components=1, backend=backend)
    x_ref = np.random.randn(100, 2)
    gmm_detector.fit(x_ref)
    gmm_detector.infer_threshold(x_ref, 0.1)
    x = np.array([[0, 10], [0, 0.1]])
    y = gmm_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 5
    assert y['instance_score'][1] < 2
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


@pytest.mark.parametrize('backend', ['pytorch', 'sklearn'])
def test_gmm_integration(backend):
    """
    Tests gmm detector on the moons dataset. Fits and infers thresholds and
    verifies that the detector can correctly detect inliers and outliers.
    """
    gmm_detector = GMM(n_components=8, backend=backend)
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    gmm_detector.fit(X_ref)
    gmm_detector.infer_threshold(X_ref, 0.1)
    result = gmm_detector.predict(x_inlier)
    result = result['data']['is_outlier'][0]
    assert not result

    x_outlier = np.array([[-1, 1.5]])
    result = gmm_detector.predict(x_outlier)
    result = result['data']['is_outlier'][0]
    assert result


def test_gmm_torchscript():
    """
    Tests gmm detector fitted on the moons dataset can be torchscripted correctly.
    """
    gmm_detector = GMM(n_components=8, backend='pytorch')
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    gmm_detector.fit(X_ref)
    gmm_detector.infer_threshold(X_ref, 0.1)
    x_outlier = np.array([[-1, 1.5]])
    ts_gmm = torch.jit.script(gmm_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = ts_gmm(x)
    assert torch.all(y == torch.tensor([False, True]))
