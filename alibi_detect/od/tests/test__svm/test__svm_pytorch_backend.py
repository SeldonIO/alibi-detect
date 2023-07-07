import pytest
import numpy as np
import torch

from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.od.pytorch.svm import BgdSVMTorch, SgdSVMTorch
from alibi_detect.exceptions import NotFittedError, ThresholdNotInferredError


@pytest.mark.parametrize('backend_cls', [BgdSVMTorch, SgdSVMTorch])
def test_svm_pytorch_scoring(backend_cls):
    """Test SVM detector pytorch scoring method.

    Tests the scoring method of the SVMTorch pytorch backend detector.
    """
    sigma = torch.tensor(2)
    svm_torch = backend_cls(
        n_components=100,
        kernel=GaussianRBF(sigma=sigma),
        nu=0.1
    )
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    svm_torch.fit(x_ref)

    x_1 = torch.tensor(np.array([[8., 8.]]))
    scores_1 = svm_torch.score(x_1)

    x_2 = torch.tensor(np.array([[13., 13.]]))
    scores_2 = svm_torch.score(x_2)

    x_3 = torch.tensor(np.array([[-100., 100.]]))
    scores_3 = svm_torch.score(x_3)

    # test correct ordering of scores given relative outlyingness of data
    assert scores_1 < scores_2 < scores_3

    # test that detector correctly detects true outlier
    svm_torch.infer_threshold(x_ref, 0.01)
    x = torch.cat((x_1, x_2, x_3))
    outputs = svm_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, True, True]))
    assert torch.all(svm_torch(x) == torch.tensor([False, True, True]))

    # test that 0.01 of the in distribution data is flagged as outliers
    x = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    outputs = svm_torch.predict(x)
    assert (outputs.is_outlier.sum()/1000) - 0.01 < 0.01


@pytest.mark.skip(reason="Can't convert GaussianRBF to torchscript due to torchscript type constraints")
@pytest.mark.parametrize('backend_cls', [BgdSVMTorch, SgdSVMTorch])
def test_svm_torch_backend_ts(tmp_path, backend_cls):
    """Test SVM detector backend is torch-scriptable and savable."""
    svm_torch = backend_cls(n_components=10, kernel=GaussianRBF())
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    svm_torch.fit(x_ref, nu=0.01)
    svm_torch.infer_threshold(x_ref, 0.1)
    pred_1 = svm_torch(x)

    svm_torch = torch.jit.script(svm_torch)
    pred_2 = svm_torch(x)
    assert torch.all(pred_1 == pred_2)

    svm_torch.save(tmp_path / 'svm_torch.pt')
    svm_torch = torch.load(tmp_path / 'svm_torch.pt')
    pred_2 = svm_torch(x)
    assert torch.all(pred_1 == pred_2)


@pytest.mark.parametrize('backend_cls', [BgdSVMTorch, SgdSVMTorch])
def test_svm_pytorch_backend_fit_errors(backend_cls):
    """Test SVM detector pytorch backend fit errors.

    Tests the correct errors are raised when using the SVMTorch pytorch backend detector.
    """
    svm_torch = backend_cls(n_components=100, kernel=GaussianRBF(), nu=0.1)
    assert not svm_torch.fitted

    # Test that the backend raises an error if it is not fitted before
    # calling forward method.
    x = torch.tensor(np.random.randn(1, 10))
    with pytest.raises(NotFittedError) as err:
        svm_torch(x)
    assert str(err.value) == f'{backend_cls.__name__} has not been fit!'

    # Test that the backend raises an error if it is not fitted before
    # predicting.
    with pytest.raises(NotFittedError) as err:
        svm_torch.predict(x)
    assert str(err.value) == f'{backend_cls.__name__} has not been fit!'

    # Test the backend updates _fitted flag on fit.
    x_ref = torch.tensor(np.random.randn(1024, 10))
    svm_torch.fit(x_ref)
    assert svm_torch.fitted

    # Test that the backend raises an if the forward method is called without the
    # threshold being inferred.
    with pytest.raises(ThresholdNotInferredError) as err:
        svm_torch(x)
    assert str(err.value) == f'{backend_cls.__name__} has no threshold set, call `infer_threshold` to fit one!'

    # Test that the backend can call predict without the threshold being inferred.
    assert svm_torch.predict(x)


@pytest.mark.parametrize('backend_cls', [BgdSVMTorch, SgdSVMTorch])
def test_svm_pytorch_fit(backend_cls):
    """Test SVM detector pytorch fit method.

    Tests pytorch detector checks for convergence and stops early if it does.
    """
    kernel = GaussianRBF(torch.tensor(1.))
    svm_torch = backend_cls(n_components=1, kernel=kernel, nu=0.01)
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    fit_results = svm_torch.fit(x_ref, tol=0.01)
    assert fit_results['converged']
    assert fit_results['n_iter'] < 100
    assert fit_results.get('lower_bound', 0) < 1


def test_sgd_bgd_diffs():
    n_components = 300
    bgd_svm = BgdSVMTorch(n_components=n_components, kernel=GaussianRBF(sigma=torch.tensor(2)), nu=0.05)
    sgd_svm = SgdSVMTorch(n_components=n_components, kernel=GaussianRBF(sigma=torch.tensor(2)), nu=0.05)

    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    bgd_svm.fit(x_ref)
    sgd_svm.fit(x_ref)

    test_x = x_ref[:1000]
    diffs = (sgd_svm.score(test_x) - bgd_svm.score(test_x)).numpy()
    assert np.abs(diffs.mean()) < 0.1
