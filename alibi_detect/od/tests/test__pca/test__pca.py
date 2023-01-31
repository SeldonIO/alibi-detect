import pytest
import numpy as np
import torch

from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.od._pca import PCA
from alibi_detect.od.base import NotFitException
from sklearn.datasets import make_moons


def make_PCA_detector(kernel=False):
    if kernel:
        pca_detector = PCA(n_components=2, kernel=GaussianRBF())
    else:
        pca_detector = PCA(n_components=2)
    x_ref = np.random.randn(100, 3)
    pca_detector.fit(x_ref)
    pca_detector.infer_threshold(x_ref, 0.1)
    return pca_detector


@pytest.mark.parametrize('detector', [
    lambda: PCA(n_components=5),
    lambda: PCA(n_components=5, kernel=GaussianRBF())
])
def test_unfitted_PCA_single_score(detector):
    pca = detector()
    x = np.array([[0, 10, 0], [0.1, 0, 0]])
    with pytest.raises(NotFitException) as err:
        _ = pca.predict(x)
    assert str(err.value) == \
        f'{pca.backend.__class__.__name__} has not been fit!'


def test_fitted_PCA_single_score():
    pca_detector = PCA(n_components=2)
    x_ref = np.random.randn(100, 3)
    pca_detector.fit(x_ref)
    x = np.array([[0, 10, 0], [0.1, 0, 0]])
    y = pca_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 5
    assert y['instance_score'][1] < 1
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


def test_fitted_kernel_PCA_single_score():
    pca_detector = PCA(n_components=2, kernel=GaussianRBF())
    x_ref = np.random.randn(100, 3) * np.array([1, 10, 0.1])
    pca_detector.fit(x_ref)
    x = np.array([[0, 5, 10], [0.1, 5, 0]])
    y = pca_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > y['instance_score'][1]
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


def test_fitted_PCA_predict():
    pca_detector = make_PCA_detector()
    x_ref = np.random.randn(100, 3)
    pca_detector.infer_threshold(x_ref, 0.1)
    x = np.array([[0, 10, 0], [0.1, 0, 0]])
    y = pca_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 5
    assert y['instance_score'][1] < 1
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


def test_fitted_kernel_PCA_predict():
    pca_detector = PCA(n_components=2, kernel=GaussianRBF())
    x_ref = np.random.randn(100, 3) * np.array([1, 10, 0.1])
    pca_detector.fit(x_ref)
    pca_detector.infer_threshold(x_ref, 0.1)
    x = np.array([[0, 5, 10], [0.1, 5, 0]])
    y = pca_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > y['instance_score'][1]
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


def test_PCA_integration():
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


def test_PCA_integration_ts():
    pca_detector = PCA(n_components=1)
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    pca_detector.fit(X_ref)
    pca_detector.infer_threshold(X_ref, 0.1)
    x_outlier = np.array([[0, -3]])
    ts_PCA = torch.jit.script(pca_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = ts_PCA(x)
    assert torch.all(y == torch.tensor([False, True]))


def test_kernel_PCA_integration():
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
