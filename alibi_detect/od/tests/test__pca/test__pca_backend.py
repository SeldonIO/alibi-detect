import pytest
import torch
import numpy as np

from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.od.pytorch.pca import LinearPCATorch, KernelPCATorch
from alibi_detect.exceptions import NotFittedError, ThresholdNotInferredError


@pytest.mark.parametrize('backend_detector', [
    lambda: LinearPCATorch(n_components=5),
    lambda: KernelPCATorch(n_components=5, kernel=GaussianRBF())
])
def test_pca_torch_backend_fit_errors(backend_detector):
    """Test Linear and Kernel PCA detector backend fit errors.

    Test that an unfit detector backend raises an error when calling predict or score. Test that the
    detector backend raises an error when calling the forward method while the threshold has not been
    inferred.
    """
    pca_torch = backend_detector()
    assert not pca_torch.fitted

    x = torch.randn((1, 10))
    with pytest.raises(NotFittedError) as err:
        pca_torch(x)
    assert str(err.value) == f'{pca_torch.__class__.__name__} has not been fit!'

    with pytest.raises(NotFittedError) as err:
        pca_torch.predict(x)
    assert str(err.value) == f'{pca_torch.__class__.__name__} has not been fit!'

    x_ref = torch.randn((1024, 10))
    pca_torch.fit(x_ref)

    assert pca_torch.fitted

    with pytest.raises(ThresholdNotInferredError) as err:
        pca_torch(x)

    assert str(err.value) == (f'{pca_torch.__class__.__name__} has no threshold set, '
                              'call `infer_threshold` to fit one!')

    assert pca_torch.predict(x)


@pytest.mark.parametrize('backend_detector', [
    lambda: LinearPCATorch(n_components=1),
    lambda: KernelPCATorch(n_components=1, kernel=GaussianRBF())
])
def test_pca_scoring(backend_detector):
    """Test Linear and Kernel PCATorch detector backend scoring methods.

    Test that the detector correctly detects true outliers and that the correct proportion of in
    distribution data is flagged as outliers.
    """
    pca_torch = backend_detector()
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    pca_torch.fit(x_ref)

    x_1 = torch.tensor([[8., 8.]], dtype=torch.float64)
    scores_1 = pca_torch.score(x_1)

    x_2 = torch.tensor([[10., 8.]], dtype=torch.float64)
    scores_2 = pca_torch.score(x_2)

    x_3 = torch.tensor([[8., 20.]], dtype=torch.float64)
    scores_3 = pca_torch.score(x_3)

    # test correct ordering of scores given outlyingness of data
    assert scores_1 < scores_2 < scores_3

    # test that detector correctly detects true Outlier
    pca_torch.infer_threshold(x_ref, 0.01)
    x = torch.cat((x_1, x_2, x_3))
    outputs = pca_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(pca_torch(x) == torch.tensor([False, False, True]))

    # test that 0.01 of the in distribution data is flagged as outliers
    x = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    outputs = pca_torch.predict(x)
    assert (outputs.is_outlier.sum()/1000) - 0.01 < 0.005


def test_pca_linear_torch_backend_ts(tmp_path):
    """Test Linear PCA detector backend is torch-scriptable and savable."""
    pca_torch = LinearPCATorch(n_components=5)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    pca_torch.fit(x_ref)
    pca_torch.infer_threshold(x_ref, 0.1)
    pred_1 = pca_torch(x)

    pca_torch = torch.jit.script(pca_torch)
    pred_2 = pca_torch(x)
    assert torch.all(pred_1 == pred_2)

    pca_torch.save(tmp_path / 'pca_torch.pt')
    pca_torch = torch.load(tmp_path / 'pca_torch.pt')
    pred_2 = pca_torch(x)
    assert torch.all(pred_1 == pred_2)


@pytest.mark.skip(reason='GaussianRBF kernel does not have torchscript support yet.')
def test_pca_kernel_torch_backend_ts(tmp_path):
    """Test Kernel PCA detector backend is torch-scriptable and savable."""
    pca_torch = KernelPCATorch(n_components=5, kernel=GaussianRBF())
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    pca_torch.fit(x_ref)
    pca_torch.infer_threshold(x_ref, 0.1)
    pred_1 = pca_torch(x)

    pca_torch = torch.jit.script(pca_torch)
    pred_2 = pca_torch(x)
    assert torch.all(pred_1 == pred_2)

    pca_torch.save(tmp_path / 'pca_torch.pt')
    pca_torch = torch.load(tmp_path / 'pca_torch.pt')
    pred_2 = pca_torch(x)
    assert torch.all(pred_1 == pred_2)
