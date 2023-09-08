import pytest
import numpy as np
import torch

from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.od._pca import PCA
from alibi_detect.exceptions import NotFittedError
from sklearn.datasets import make_moons


def fit_PCA_detector(detector):
    pca_detector = detector()
    x_ref = np.random.randn(100, 3)
    pca_detector.fit(x_ref)
    pca_detector.infer_threshold(x_ref, 0.1)
    return pca_detector


@pytest.mark.parametrize('detector', [
    lambda: PCA(n_components=3),
    lambda: PCA(n_components=3, kernel=GaussianRBF())
])
def test_unfitted_PCA_single_score(detector):
    """Test pca detector throws errors when not fitted."""
    pca = detector()
    x = np.array([[0, 10, 11], [0.1, 0, 11]])
    x_ref = np.random.randn(100, 3)

    # test infer_threshold raises exception when not fitted
    with pytest.raises(NotFittedError) as err:
        pca.infer_threshold(x_ref, 0.1)
    assert str(err.value) == \
        f'{pca.__class__.__name__} has not been fit!'

    # test score raises exception when not fitted
    with pytest.raises(NotFittedError) as err:
        pca.score(x)
    assert str(err.value) == \
        f'{pca.__class__.__name__} has not been fit!'

    # test predict raises exception when not fitted
    with pytest.raises(NotFittedError) as err:
        pca.predict(x)
    assert str(err.value) == \
        f'{pca.__class__.__name__} has not been fit!'


def test_pca_value_errors():
    with pytest.raises(ValueError) as err:
        PCA(n_components=0)
    assert str(err.value) == 'n_components must be at least 1'

    with pytest.raises(ValueError) as err:
        pca = PCA(n_components=4)
        pca.fit(np.random.randn(100, 3))
    assert str(err.value) == 'n_components must be less than the number of features.'

    with pytest.raises(ValueError) as err:
        pca = PCA(n_components=10, kernel=GaussianRBF())
        pca.fit(np.random.randn(9, 3))
    assert str(err.value) == 'n_components must be less than the number of reference instances.'


@pytest.mark.parametrize('detector', [
    lambda: PCA(n_components=2),
    lambda: PCA(n_components=2, kernel=GaussianRBF())
])
def test_fitted_PCA_score(detector):
    """Test Linear and Kernel PCA detector score method.

    Test Linear and Kernel PCA detector that has been fitted on reference data but has not had a threshold
    inferred can still score data using the predict method. Test that it does not raise an error
    and does not return `threshold`, `p_value` and `is_outlier` values.
    """
    pca_detector = detector()
    x_ref = np.random.randn(100, 3)
    pca_detector.fit(x_ref)
    x = np.array([[0, 10, 0], [0.1, 0, 0]])
    y = pca_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > y['instance_score'][1]
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


@pytest.mark.parametrize('detector', [
    lambda: PCA(n_components=2),
    lambda: PCA(n_components=2, kernel=GaussianRBF())
])
def test_fitted_PCA_predict(detector):
    """Test Linear and Kernel PCA detector predict method.

    Test Linear and Kernel PCA detector that has been fitted on reference data and has had a threshold
    inferred can score data using the predict method. Test that it does not raise an error and does
    return `threshold`, `p_value` and `is_outlier` values.
    """
    pca_detector = fit_PCA_detector(detector)
    x_ref = np.random.randn(100, 3)
    pca_detector.infer_threshold(x_ref, 0.1)
    x = np.array([[0, 10, 0], [0.1, 0, 0]])
    y = pca_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > y['instance_score'][1]
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert isinstance(y['threshold'], float)
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


def test_PCA_integration(tmp_path):
    """Test Linear PCA detector on moons dataset.

    Test the Linear PCA detector on a more complex 2d example. Test that the detector can be fitted
    on reference data and infer a threshold. Test that it differentiates between inliers and outliers.
    Test that the detector can be scripted.
    """
    pca_detector = PCA(n_components=1)
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    pca_detector.fit(X_ref)
    pca_detector.infer_threshold(X_ref, 0.1)
    result = pca_detector.predict(x_inlier)
    result = result['data']['is_outlier'][0]
    assert not result

    x_outlier = np.array([[0, -3]])
    result = pca_detector.predict(x_outlier)
    result = result['data']['is_outlier'][0]
    assert result

    ts_PCA = torch.jit.script(pca_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = ts_PCA(x)
    assert torch.all(y == torch.tensor([False, True]))

    ts_PCA.save(tmp_path / 'pca.pt')
    pca_detector = PCA(n_components=1)
    pca_detector = torch.load(tmp_path / 'pca.pt')
    y = pca_detector(x)
    assert torch.all(y == torch.tensor([False, True]))


def test_kernel_PCA_integration():
    """Test kernel PCA detector on moons dataset.

    Test the kernel PCA detector on a more complex 2d example. Test that the detector can be fitted
    on reference data and infer a threshold. Test that it differentiates between inliers and outliers.
    """
    pca_detector = PCA(n_components=10, kernel=GaussianRBF())
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    pca_detector.fit(X_ref)
    pca_detector.infer_threshold(X_ref, 0.1)
    result = pca_detector.predict(x_inlier)
    result = result['data']['is_outlier'][0]
    assert not result

    x_outlier = np.array([[1, 1]])
    result = pca_detector.predict(x_outlier)
    result = result['data']['is_outlier'][0]
    assert result


@pytest.mark.skip(reason='GaussianRBF kernel does not have torchscript support yet.')
def test_kernel_PCA_integration_ts():
    """Test the kernel PCA detector can be scripted."""
    pca_detector = PCA(n_components=10, kernel=GaussianRBF())
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    pca_detector.fit(X_ref)
    pca_detector.infer_threshold(X_ref, 0.1)
    x_outlier = np.array([[1, 1]])
    ts_PCA = torch.jit.script(pca_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = ts_PCA(x)
    assert torch.all(y == torch.tensor([False, True]))
