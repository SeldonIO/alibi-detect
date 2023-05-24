import pytest
import torch

from alibi_detect.od.pytorch.knn import KNNTorch
from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.od.pytorch.ensemble import Ensembler, PValNormalizer, AverageAggregator
from alibi_detect.exceptions import NotFittedError, ThresholdNotInferredError


@pytest.fixture(scope='function')
def ensembler(request):
    return Ensembler(
        normalizer=PValNormalizer(),
        aggregator=AverageAggregator()
    )


def test_knn_torch_backend():
    """
    Test the knn torch backend can be correctly initialized, fit and used to
    predict outliers.
    """

    knn_torch = KNNTorch(k=5)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    outputs = knn_torch.predict(x)
    assert outputs.instance_score.shape == (3, )
    assert outputs.is_outlier is None
    assert outputs.p_value is None
    scores = knn_torch.score(x)
    assert torch.all(scores == outputs.instance_score)

    knn_torch.infer_threshold(x_ref, 0.1)
    outputs = knn_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(knn_torch(x) == torch.tensor([False, False, True]))


def test_knn_torch_backend_ensemble(ensembler):
    """
    Test the knn torch backend can be correctly initialized as an ensemble, fit
    on data and used to predict outliers.
    """

    knn_torch = KNNTorch(k=[4, 5], ensembler=ensembler)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    knn_torch.infer_threshold(x_ref, 0.1)
    outputs = knn_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(knn_torch(x) == torch.tensor([False, False, True]))


def test_knn_torch_backend_ensemble_ts(tmp_path, ensembler):
    """
    Test the knn torch backend can be initialized as an ensemble and
    torchscripted, as well as saved and loaded to and from disk.
    """

    knn_torch = KNNTorch(k=[4, 5], ensembler=ensembler)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    knn_torch.infer_threshold(x_ref, 0.1)
    pred_1 = knn_torch(x)
    knn_torch = torch.jit.script(knn_torch)
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)

    knn_torch.save(tmp_path / 'knn_torch.pt')
    knn_torch = torch.load(tmp_path / 'knn_torch.pt')
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)


def test_knn_torch_backend_ts(tmp_path):
    """
    Test the knn torch backend can be initialized and torchscripted, as well as
    saved and loaded to and from disk.
    """

    knn_torch = KNNTorch(k=7)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    knn_torch.infer_threshold(x_ref, 0.1)
    pred_1 = knn_torch(x)
    knn_torch = torch.jit.script(knn_torch)
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)

    knn_torch.save(tmp_path / 'knn_torch.pt')
    knn_torch = torch.load(tmp_path / 'knn_torch.pt')
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)


def test_knn_kernel(ensembler):
    """
    Test the knn torch backend can be correctly initialized with a kernel, fit
    on data and used to predict outliers.
    """

    kernel = GaussianRBF(sigma=torch.tensor((0.25)))
    knn_torch = KNNTorch(k=[4, 5], kernel=kernel, ensembler=ensembler)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    knn_torch.infer_threshold(x_ref, 0.1)
    outputs = knn_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(knn_torch(x) == torch.tensor([False, False, True]))


@pytest.mark.skip(reason="Can't convert GaussianRBF to torchscript due to torchscript type constraints")
def test_knn_kernel_ts(ensembler):
    """
    Test the knn torch backend can be correctly initialized with a kernel,
    and torchscripted, as well as saved and loaded to and from disk.
    """

    kernel = GaussianRBF(sigma=torch.tensor((0.25)))
    knn_torch = KNNTorch(k=[4, 5], kernel=kernel, ensembler=ensembler)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    knn_torch.infer_threshold(x_ref, 0.1)
    pred_1 = knn_torch(x)
    knn_torch = torch.jit.script(knn_torch)
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)


def test_knn_torch_backend_ensemble_fit_errors(ensembler):
    """Tests the correct errors are raised when using the KNNTorch backend as an ensemble."""
    knn_torch = KNNTorch(k=[4, 5], ensembler=ensembler)

    # Test that the backend raises an error if it is not fitted before
    # calling forward method.
    x = torch.randn((1, 10))
    with pytest.raises(NotFittedError) as err:
        knn_torch(x)
    assert str(err.value) == 'KNNTorch has not been fit!'

    # Test that the backend raises an error if it is not fitted before
    # predicting.
    with pytest.raises(NotFittedError) as err:
        knn_torch.predict(x)
    assert str(err.value) == 'KNNTorch has not been fit!'

    # Test the backend updates fitted flag on fit.
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    assert knn_torch.fitted

    # Test that the backend raises an if the forward method is called without the
    # threshold being inferred.
    with pytest.raises(ThresholdNotInferredError) as err:
        knn_torch(x)
    assert str(err.value) == 'KNNTorch has no threshold set, call `infer_threshold` to fit one!'

    # Test that the backend can call predict without the threshold being inferred.
    with pytest.raises(ThresholdNotInferredError) as err:
        knn_torch.predict(x)
    assert str(err.value) == 'KNNTorch has no threshold set, call `infer_threshold` to fit one!'


def test_knn_torch_backend_fit_errors():
    """Tests the correct errors are raised when using the KNNTorch backend as a single detector."""
    knn_torch = KNNTorch(k=4)

    # Test that the backend raises an error if it is not fitted before
    # calling forward method.
    x = torch.randn((1, 10))
    with pytest.raises(NotFittedError) as err:
        knn_torch(x)
    assert str(err.value) == 'KNNTorch has not been fit!'

    # Test that the backend raises an error if it is not fitted before
    # predicting.
    with pytest.raises(NotFittedError) as err:
        knn_torch.predict(x)
    assert str(err.value) == 'KNNTorch has not been fit!'

    # Test the backend updates fitted flag on fit.
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    assert knn_torch.fitted

    # Test that the backend raises an if the forward method is called without the
    # threshold being inferred.
    with pytest.raises(ThresholdNotInferredError) as err:
        knn_torch(x)
    assert str(err.value) == 'KNNTorch has no threshold set, call `infer_threshold` to fit one!'

    # Test that the backend can call predict without the threshold being inferred.
    knn_torch.predict(x)


def test_knn_infer_threshold_value_errors():
    """Tests the correct errors are raised when using incorrect choice of fpr for the KNNTorch backend detector."""
    knn_torch = KNNTorch(k=4)
    x = torch.randn((1024, 10))
    knn_torch.fit(x)

    # fpr must be greater than 1/len(x) otherwise it excludes all points in the reference dataset
    with pytest.raises(ValueError) as err:
        knn_torch.infer_threshold(x, 1/1025)
    assert str(err.value) == '`fpr` must be greater than `1/len(x)=0.0009765625`.'

    # fpr must be between 0 and 1
    with pytest.raises(ValueError) as err:
        knn_torch.infer_threshold(x, 1.1)
    assert str(err.value) == '`fpr` must be in `(0, 1)`.'

    knn_torch.infer_threshold(x, 0.99)
    knn_torch.infer_threshold(x,  1/1023)
