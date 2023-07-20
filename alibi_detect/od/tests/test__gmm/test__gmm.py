import pytest
import numpy as np
import torch

from alibi_detect.od._gmm import GMM
from alibi_detect.exceptions import NotFittedError

from sklearn.datasets import make_moons


@pytest.mark.parametrize('backend', ['pytorch', 'sklearn'])
def test_unfitted_gmm_score(backend):
    """Test GMM detector raises exceptions when not fitted."""
    gmm_detector = GMM(n_components=2, backend=backend)
    x = np.array([[0, 10], [0.1, 0]])
    x_ref = np.random.randn(100, 2)

    with pytest.raises(NotFittedError) as err:
        gmm_detector.infer_threshold(x_ref, 0.1)
    assert str(err.value) == 'GMM has not been fit!'

    with pytest.raises(NotFittedError) as err:
        gmm_detector.score(x)
    assert str(err.value) == 'GMM has not been fit!'

    # test predict raises exception when not fitted
    with pytest.raises(NotFittedError) as err:
        gmm_detector.predict(x)
    assert str(err.value) == 'GMM has not been fit!'


@pytest.mark.parametrize('backend', ['pytorch', 'sklearn'])
def test_fitted_gmm_score(backend):
    """Test GMM detector score method.

    Test GMM detector that has been fitted on reference data but has not had a threshold
    inferred can still score data using the predict method. Test that it does not raise an error
    but does not return `threshold`, `p_value` and `is_outlier` values.
    """
    gmm_detector = GMM(n_components=1, backend=backend)
    x_ref = np.random.randn(100, 2)
    gmm_detector.fit(x_ref)
    x = np.array([[0, 10], [0.1, 0]])
    scores = gmm_detector.score(x)

    y = gmm_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 5
    assert y['instance_score'][1] < 2
    assert all(y['instance_score'] == scores)
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


@pytest.mark.parametrize('backend', ['pytorch', 'sklearn'])
def test_fitted_gmm_predict(backend):
    """Test GMM detector predict method.

    Test GMM detector that has been fitted on reference data and has had a threshold
    inferred can score data using the predict method as well as predict outliers. Test that it
    returns `threshold`, `p_value` and `is_outlier` values.
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
    assert isinstance(y['threshold'], float)
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


@pytest.mark.parametrize('backend', ['pytorch', 'sklearn'])
def test_gmm_integration(backend):
    """Test GMM detector on moons dataset.

    Test GMM detector on a more complex 2d example. Test that the detector can be fitted
    on reference data and infer a threshold. Test that it differentiates between inliers and outliers.
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


def test_gmm_torchscript(tmp_path):
    """Tests user can torch-script gmm detector."""
    gmm_detector = GMM(n_components=8, backend='pytorch')
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    gmm_detector.fit(X_ref)
    gmm_detector.infer_threshold(X_ref, 0.1)
    x_outlier = np.array([[-1, 1.5]])
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)

    ts_gmm = torch.jit.script(gmm_detector.backend)
    y = ts_gmm(x)
    assert torch.all(y == torch.tensor([False, True]))

    ts_gmm.save(tmp_path / 'gmm.pt')
    ts_gmm = torch.load(tmp_path / 'gmm.pt')
    y = ts_gmm(x)
    assert torch.all(y == torch.tensor([False, True]))


@pytest.mark.parametrize('backend', ['pytorch', 'sklearn'])
def test_gmm_fit(backend):
    """Test GMM detector fit method.

    Tests detector checks for convergence and stops early if it does.
    """
    gmm = GMM(n_components=1, backend=backend)
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    fit_results = gmm.fit(x_ref, tol=0.01, batch_size=32)
    assert isinstance(fit_results['lower_bound'], float)
    assert fit_results['converged']
    assert fit_results['lower_bound'] < 1
