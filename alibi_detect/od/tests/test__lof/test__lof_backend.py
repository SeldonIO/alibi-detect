import pytest
import torch

from alibi_detect.od.pytorch.lof import LOFTorch
from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.od.pytorch.ensemble import Ensembler, PValNormalizer, AverageAggregator
from alibi_detect.exceptions import NotFittedError, ThresholdNotInferredError


@pytest.fixture(scope='session')
def ensembler(request):
    return Ensembler(
        normalizer=PValNormalizer(),
        aggregator=AverageAggregator()
    )


def test_lof_torch_backend():
    """
    Test the lof torch backend can be correctly initialized, fit and used to
    predict outliers.
    """

    lof_torch = LOFTorch(k=5)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    lof_torch.fit(x_ref)
    outputs = lof_torch.predict(x)
    assert outputs.instance_score.shape == (3, )
    assert outputs.is_outlier is None
    assert outputs.p_value is None
    scores = lof_torch.score(x)
    assert torch.all(scores == outputs.instance_score)

    lof_torch.infer_threshold(x_ref, 0.1)
    outputs = lof_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(lof_torch(x) == torch.tensor([False, False, True]))


def test_lof_torch_backend_ensemble(ensembler):
    """
    Test the lof torch backend can be correctly initialized as an ensemble, fit
    on data and used to predict outliers.
    """

    lof_torch = LOFTorch(k=[4, 5], ensembler=ensembler)
    x_ref = torch.randn((1024, 10))
    lof_torch.fit(x_ref)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    result = lof_torch.predict(x)
    assert result.instance_score.shape == (3, )

    lof_torch.infer_threshold(x_ref, 0.1)
    outputs = lof_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(lof_torch(x) == torch.tensor([False, False, True]))


def test_lof_torch_backend_ensemble_ts(tmp_path, ensembler):
    """
    Test the lof torch backend can be initalized as an ensemble and
    torchscripted, as well as saved and loaded to and from disk.
    """

    lof_torch = LOFTorch(k=[4, 5], ensembler=ensembler)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])

    with pytest.raises(NotFittedError) as err:
        lof_torch(x)
    assert str(err.value) == 'LOFTorch has not been fit!'

    with pytest.raises(NotFittedError) as err:
        lof_torch.predict(x)
    assert str(err.value) == 'LOFTorch has not been fit!'

    x_ref = torch.randn((1024, 10))
    lof_torch.fit(x_ref)
    lof_torch.infer_threshold(x_ref, 0.1)
    pred_1 = lof_torch(x)
    lof_torch = torch.jit.script(lof_torch)
    pred_2 = lof_torch(x)
    assert torch.all(pred_1 == pred_2)

    lof_torch.save(tmp_path / 'lof_torch.pt')
    lof_torch = torch.load(tmp_path / 'lof_torch.pt')
    pred_2 = lof_torch(x)
    assert torch.all(pred_1 == pred_2)


def test_lof_torch_backend_ts(tmp_path):
    """
    Test the lof torch backend can be initalized and torchscripted, as well as
    saved and loaded to and from disk.
    """

    lof_torch = LOFTorch(k=7)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    lof_torch.fit(x_ref)
    lof_torch.infer_threshold(x_ref, 0.1)
    pred_1 = lof_torch(x)
    lof_torch = torch.jit.script(lof_torch)
    pred_2 = lof_torch(x)
    assert torch.all(pred_1 == pred_2)

    lof_torch.save(tmp_path / 'lof_torch.pt')
    lof_torch = torch.load(tmp_path / 'lof_torch.pt')
    pred_2 = lof_torch(x)
    assert torch.all(pred_1 == pred_2)


def test_lof_kernel(ensembler):
    """
    Test the lof torch backend can be correctly initialized with a kernel, fit
    on data and used to predict outliers.
    """

    kernel = GaussianRBF(sigma=torch.tensor((1)))
    lof_torch = LOFTorch(k=[4, 5], kernel=kernel, ensembler=ensembler)
    x_ref = torch.randn((1024, 10))
    lof_torch.fit(x_ref)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    result = lof_torch.predict(x)
    assert result.instance_score.shape == (3,)

    lof_torch.infer_threshold(x_ref, 0.1)
    outputs = lof_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(lof_torch(x) == torch.tensor([False, False, True]))


@pytest.mark.skip(reason="Can't convert GaussianRBF to torchscript due to torchscript type constraints")
def test_lof_kernel_ts(ensembler):
    """
    Test the lof torch backend can be correctly initialized with a kernel,
    and torchscripted, as well as saved and loaded to and from disk.
    """

    kernel = GaussianRBF(sigma=torch.tensor((0.25)))
    lof_torch = LOFTorch(k=[4, 5], kernel=kernel, ensembler=ensembler)
    x_ref = torch.randn((1024, 10))
    lof_torch.fit(x_ref)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    lof_torch.infer_threshold(x_ref, 0.1)
    pred_1 = lof_torch(x)
    lof_torch = torch.jit.script(lof_torch)
    pred_2 = lof_torch(x)
    assert torch.all(pred_1 == pred_2)


@pytest.mark.parametrize('k', [[4, 5], 4])
def test_lof_torch_backend_ensemble_fit_errors(k, ensembler):
    lof_torch = LOFTorch(k=[4, 5], ensembler=ensembler)
    assert not lof_torch._fitted

    # Test that the backend raises an error if it is not fitted before
    # calling forward method.
    x = torch.randn((1, 10))
    with pytest.raises(NotFittedError) as err:
        lof_torch(x)
    assert str(err.value) == 'LOFTorch has not been fit!'

    # Test that the backend raises an error if it is not fitted before
    # predicting.
    with pytest.raises(NotFittedError) as err:
        lof_torch.predict(x)
    assert str(err.value) == 'LOFTorch has not been fit!'

    # Test the backend updates _fitted flag on fit.
    x_ref = torch.randn((1024, 10))
    lof_torch.fit(x_ref)
    assert lof_torch._fitted

    # Test that the backend raises an if the forward method is called without the
    # threshold being inferred.
    with pytest.raises(ThresholdNotInferredError) as err:
        lof_torch(x)
    assert str(err.value) == 'LOFTorch has no threshold set, call `infer_threshold` before predicting.'

    # Test that the backend can call predict without the threshold being inferred.
    assert lof_torch.predict(x)
