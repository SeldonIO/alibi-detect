import pytest
import torch

from alibi_detect.od.backend.torch.knn import KNNTorch
# from alibi_detect.utils.pytorch.kernels import GaussianRBF


def test_knn_torch_backend():
    knn_torch = KNNTorch(k=5)
    x = torch.randn((3, 10))

    with pytest.raises(AttributeError):
        # TODO: should be a different error!
        knn_torch(x)

    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    outputs = knn_torch.predict(x)
    assert outputs['scores'].shape == (3, )
    assert outputs['preds'] is None
    assert outputs['p_vals'] is None
    scores = knn_torch.score(x)
    assert torch.all(scores == outputs['scores'])

    knn_torch.infer_threshold(x_ref, 0.1)
    outputs = knn_torch.predict(x)
    assert torch.all(outputs['preds'] == torch.tensor([False, False, False]))

    x = torch.randn((1, 10)) * 100
    assert knn_torch(x)


def test_knn_torch_backend_ensemble(accumulator):
    knn_torch = KNNTorch(k=[4, 5], accumulator=accumulator)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10))
    result = knn_torch.predict(x)
    assert result['scores'].shape == (3, )

    knn_torch.infer_threshold(x_ref, 0.1)
    outputs = knn_torch.predict(x)
    assert torch.all(outputs['preds'] == torch.tensor([False, False, False]))

    x = torch.randn((1, 10)) * 100
    assert knn_torch(x)


def test_knn_torch_backend_ensemble_ts(accumulator):
    knn_torch = KNNTorch(k=[4, 5], accumulator=accumulator)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    knn_torch.infer_threshold(x_ref, 0.1)
    pred_1 = knn_torch(x)
    knn_torch = torch.jit.script(knn_torch)
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)


def test_knn_torch_backend_ts():
    knn_torch = KNNTorch(k=7)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    knn_torch.infer_threshold(x_ref, 0.1)
    pred_1 = knn_torch(x)
    knn_torch = torch.jit.script(knn_torch)
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)


# def test_knn_kernel(accumulator):
#     kernel = GaussianRBF(sigma=torch.tensor((1)))
#     knn_torch = KNNTorch(k=[4, 5], kernel=kernel, accumulator=accumulator)
#     x_ref = torch.randn((1024, 10))
#     knn_torch.fit(x_ref)
#     x = torch.randn((3, 10))
#     scores = knn_torch(x)
#     assert scores.shape == (3,)

#     knn_torch.infer_threshold(x_ref, 0.1)
#     outputs = knn_torch.predict(x)
#     assert torch.all(outputs['preds'] == torch.tensor([False, False, False]))

#     x = torch.randn((1, 10)) * 100
#     print(knn_torch.predict(x))
#     # assert knn_torch(x).item()
