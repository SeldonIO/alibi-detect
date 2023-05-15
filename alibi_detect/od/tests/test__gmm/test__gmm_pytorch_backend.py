import pytest
import numpy as np
import torch

from alibi_detect.od.pytorch.gmm import GMMTorch
from alibi_detect.exceptions import NotFittedError, ThresholdNotInferredError


def test_gmm_pytorch_scoring():
    """Test GMM detector pytorch scoring method.

    Tests the scoring method of the GMMTorch pytorch backend detector.
    """
    gmm_torch = GMMTorch(n_components=1)
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    gmm_torch.fit(x_ref)

    x_1 = torch.tensor(np.array([[8., 8.]]))
    scores_1 = gmm_torch.score(x_1)

    x_2 = torch.tensor(np.random.multivariate_normal(mean, cov, 1))
    scores_2 = gmm_torch.score(x_2)

    x_3 = torch.tensor(np.array([[-10., 10.]]))
    scores_3 = gmm_torch.score(x_3)

    # test correct ordering of scores given outlyingness of data
    assert scores_1 < scores_2 < scores_3

    # test that detector correctly detects true outlier
    gmm_torch.infer_threshold(x_ref, 0.01)
    x = torch.cat((x_1, x_2, x_3))
    outputs = gmm_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(gmm_torch(x) == torch.tensor([False, False, True]))

    # test that 0.01 of the in distribution data is flagged as outliers
    x = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    outputs = gmm_torch.predict(x)
    assert (outputs.is_outlier.sum()/1000) - 0.01 < 0.01


def test_gmm_torch_backend_ts(tmp_path):
    """Test GMM detector backend is torch-scriptable and savable."""
    gmm_torch = GMMTorch(n_components=2)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    gmm_torch.fit(x_ref)
    gmm_torch.infer_threshold(x_ref, 0.1)
    pred_1 = gmm_torch(x)

    gmm_torch = torch.jit.script(gmm_torch)
    pred_2 = gmm_torch(x)
    assert torch.all(pred_1 == pred_2)

    gmm_torch.save(tmp_path / 'gmm_torch.pt')
    gmm_torch = torch.load(tmp_path / 'gmm_torch.pt')
    pred_2 = gmm_torch(x)
    assert torch.all(pred_1 == pred_2)


def test_gmm_pytorch_backend_fit_errors():
    """Test gmm detector pytorch backend fit errors.

    Tests the correct errors are raised when using the GMMTorch pytorch backend detector.
    """
    gmm_torch = GMMTorch(n_components=2)
    assert not gmm_torch.fitted

    # Test that the backend raises an error if it is not fitted before
    # calling forward method.
    x = torch.tensor(np.random.randn(1, 10))
    with pytest.raises(NotFittedError) as err:
        gmm_torch(x)
    assert str(err.value) == 'GMMTorch has not been fit!'

    # Test that the backend raises an error if it is not fitted before
    # predicting.
    with pytest.raises(NotFittedError) as err:
        gmm_torch.predict(x)
    assert str(err.value) == 'GMMTorch has not been fit!'

    # Test the backend updates _fitted flag on fit.
    x_ref = torch.tensor(np.random.randn(1024, 10))
    gmm_torch.fit(x_ref)
    assert gmm_torch.fitted

    # Test that the backend raises an if the forward method is called without the
    # threshold being inferred.
    with pytest.raises(ThresholdNotInferredError) as err:
        gmm_torch(x)
    assert str(err.value) == 'GMMTorch has no threshold set, call `infer_threshold` to fit one!'

    # Test that the backend can call predict without the threshold being inferred.
    assert gmm_torch.predict(x)


def test_gmm_pytorch_fit():
    """Test GMM detector pytorch fit method.

    Tests pytorch detector checks for convergence and stops early if it does.
    """
    gmm_torch = GMMTorch(n_components=1)
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    fit_results = gmm_torch.fit(x_ref, tol=0.01)
    assert fit_results['converged']
    assert fit_results['n_epochs'] < 10
    assert fit_results['lower_bound'] < 1
