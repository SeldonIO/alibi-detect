import pytest
import torch
import numpy as np

from alibi_detect.od.pytorch.mahalanobis import MahalanobisTorch
from alibi_detect.base import NotFitException, ThresholdNotInferredException


def test_mahalanobis_torch_backend_fit_errors():
    mahalanobis_torch = MahalanobisTorch()
    assert not mahalanobis_torch._fitted

    x = torch.randn((1, 10))
    with pytest.raises(NotFitException) as err:
        mahalanobis_torch(x)
    assert str(err.value) == 'MahalanobisTorch has not been fit!'

    with pytest.raises(NotFitException) as err:
        mahalanobis_torch.predict(x)
    assert str(err.value) == 'MahalanobisTorch has not been fit!'

    x_ref = torch.randn((1024, 10))
    mahalanobis_torch.fit(x_ref)

    assert mahalanobis_torch._fitted

    with pytest.raises(ThresholdNotInferredException) as err:
        mahalanobis_torch(x)
    assert str(err.value) == 'MahalanobisTorch has no threshold set, call `infer_threshold` before predicting.'

    assert mahalanobis_torch.predict(x)


def test_mahalanobis_linear_scoring():
    mahalanobis_torch = MahalanobisTorch()
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    mahalanobis_torch.fit(x_ref)
    p = mahalanobis_torch._compute_linear_proj(mahalanobis_torch.x_ref)

    # test that the x_ref is whitened by the data
    assert p.mean() < 0.1
    assert p.std() - 1 < 0.1

    x_1 = torch.tensor([[8., 8.]])
    scores_1 = mahalanobis_torch.score(x_1)

    x_2 = torch.tensor(np.random.multivariate_normal(mean, cov, 1))
    scores_2 = mahalanobis_torch.score(x_2)

    x_3 = torch.tensor([[-10., 10.]])
    scores_3 = mahalanobis_torch.score(x_3)

    # test correct ordering of scores given outlyingness of data
    assert scores_1 < scores_2 < scores_3

    # test that detector correctly detects true Outlier
    mahalanobis_torch.infer_threshold(x_ref, 0.01)
    x = np.concatenate((x_1, x_2, x_3))
    outputs = mahalanobis_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(mahalanobis_torch(x) == torch.tensor([False, False, True]))

    # test that 0.01 of the in distribution data is flagged as outliers
    x = torch.tensor(np.random.multivariate_normal(mean, cov, 1000))
    outputs = mahalanobis_torch.predict(x)
    assert (outputs.is_outlier.sum()/1000) - 0.01 < 0.005


def test_mahalanobis_torch_backend_ts(tmp_path):
    mahalanobis_torch = MahalanobisTorch()
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    mahalanobis_torch.fit(x_ref)
    mahalanobis_torch.infer_threshold(x_ref, 0.1)
    pred_1 = mahalanobis_torch(x)

    mahalanobis_torch = torch.jit.script(mahalanobis_torch)
    pred_2 = mahalanobis_torch(x)
    assert torch.all(pred_1 == pred_2)

    mahalanobis_torch.save(tmp_path / 'mahalanobis_torch.pt')
    mahalanobis_torch = torch.load(tmp_path / 'mahalanobis_torch.pt')
    pred_2 = mahalanobis_torch(x)
    assert torch.all(pred_1 == pred_2)
