import pytest
import numpy as np
import torch

from alibi_detect.od.pytorch.gmm import GMMTorch
from alibi_detect.od.base import NotFitException, ThresholdNotInferredException


def test_gmm_pytorch_backend_fit_errors():
    gmm_torch = GMMTorch(n_components=2)
    assert not gmm_torch._fitted

    x = torch.tensor(np.random.randn(1, 10))
    with pytest.raises(NotFitException) as err:
        gmm_torch(x)
    assert str(err.value) == 'GMMTorch has not been fit!'

    with pytest.raises(NotFitException) as err:
        gmm_torch.predict(x)
    assert str(err.value) == 'GMMTorch has not been fit!'

    x_ref = torch.tensor(np.random.randn(1024, 10))
    gmm_torch.fit(x_ref)

    assert gmm_torch._fitted

    with pytest.raises(ThresholdNotInferredException) as err:
        gmm_torch(x)
    assert str(err.value) == 'GMMTorch has no threshold set, call `infer_threshold` before predicting.'

    assert gmm_torch.predict(x)


def test_gmm_pytorch_scoring():
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

    # test that detector correctly detects true Outlier
    gmm_torch.infer_threshold(x_ref, 0.01)
    x = torch.cat((x_1, x_2, x_3))
    outputs = gmm_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(gmm_torch(x) == torch.tensor([False, False, True]))

    # test that 0.01 of the in distribution data is flagged as outliers
    x = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    outputs = gmm_torch.predict(x)
    assert (outputs.is_outlier.sum()/1000) - 0.01 < 0.01
