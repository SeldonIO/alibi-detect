import pytest
import numpy as np
import torch

from alibi_detect.od._svm import SVM
from alibi_detect.exceptions import NotFittedError
from alibi_detect.utils.pytorch import GaussianRBF

from sklearn.datasets import make_moons


def test_unfitted_svm_score():
    """Test SVM detector raises exceptions when not fitted."""
    svm_detector = SVM(
        n_components=10,
        backend='pytorch',
        kernel=GaussianRBF()
    )
    x = np.array([[0, 10], [0.1, 0]])
    x_ref = np.random.randn(100, 2)

    with pytest.raises(NotFittedError) as err:
        svm_detector.infer_threshold(x_ref, 0.1)
    assert str(err.value) == 'SVM has not been fit!'

    with pytest.raises(NotFittedError) as err:
        svm_detector.score(x)
    assert str(err.value) == 'SVM has not been fit!'

    # test predict raises exception when not fitted
    with pytest.raises(NotFittedError) as err:
        svm_detector.predict(x)
    assert str(err.value) == 'SVM has not been fit!'


def test_fitted_svm_score():
    """Test SVM detector score method.

    Test SVM detector that has been fitted on reference data but has not had a threshold
    inferred can still score data using the predict method. Test that it does not raise an error
    but does not return `threshold`, `p_value` and `is_outlier` values.
    """
    sigma = torch.tensor(2)
    svm_detector = SVM(
        n_components=10,
        backend='pytorch',
        kernel=GaussianRBF(sigma=sigma)
    )
    x_ref = np.random.randn(100, 2)
    svm_detector.fit(x_ref, nu=0.1)
    x = np.array([[0, 10], [0.1, 0]])
    scores = svm_detector.score(x)

    y = svm_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > -0.85
    assert y['instance_score'][1] < -0.99
    assert all(y['instance_score'] == scores)
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


def test_fitted_svm_predict():
    """Test SVM detector predict method.

    Test SVM detector that has been fitted on reference data and has had a threshold
    inferred can score data using the predict method as well as predict outliers. Test that it
    returns `threshold`, `p_value` and `is_outlier` values.
    """
    sigma = torch.tensor(2)
    svm_detector = SVM(
        n_components=10,
        backend='pytorch',
        kernel=GaussianRBF(sigma=sigma)
    )
    x_ref = np.random.randn(100, 2)
    svm_detector.fit(x_ref, nu=0.1)
    svm_detector.infer_threshold(x_ref, 0.1)
    x = np.array([[0, 10], [0, 0.1]])
    y = svm_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > -0.85
    assert y['instance_score'][1] < -0.99
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


def test_svm_integration():
    """Test SVM detector on moons dataset.

    Test SVM detector on a more complex 2d example. Test that the detector can be fitted
    on reference data and infer a threshold. Test that it differentiates between inliers and outliers.
    """
    sigma = torch.tensor(0.2)
    svm_detector = SVM(
        n_components=100,
        backend='pytorch',
        kernel=GaussianRBF(sigma=sigma)
    )
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    svm_detector.fit(X_ref, nu=0.1)
    svm_detector.infer_threshold(X_ref, 0.1)
    result = svm_detector.predict(x_inlier)
    result = result['data']['is_outlier'][0]
    assert not result

    x_outlier = np.array([[-1, 1.5]])
    result = svm_detector.predict(x_outlier)
    result = result['data']['is_outlier'][0]
    assert result


@pytest.mark.skip(reason="Can't convert default kernel GaussianRBF to torchscript due to torchscript type constraints")
def test_svm_torchscript(tmp_path):
    """Tests user can torch-script svm detector."""
    sigma = torch.tensor(0.2)
    svm_detector = SVM(
        n_components=100,
        backend='pytorch',
        kernel=GaussianRBF(sigma=sigma)
    )
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    svm_detector.fit(X_ref, nu=0.1)
    svm_detector.infer_threshold(X_ref, 0.1)
    x_outlier = np.array([[-1, 1.5]])
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)

    ts_svm = torch.jit.script(svm_detector.backend)
    y = ts_svm(x)
    assert torch.all(y == torch.tensor([False, True]))

    ts_svm.save(tmp_path / 'svm.pt')
    ts_svm = torch.load(tmp_path / 'svm.pt')
    y = ts_svm(x)
    assert torch.all(y == torch.tensor([False, True]))
