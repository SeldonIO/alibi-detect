import torch

from alibi_detect.od.backend.torch.knn import KNNTorch
from alibi_detect.utils.pytorch.kernels import GaussianRBF


def test_knn_torch_backend(accumulator):
    knn_torch = KNNTorch(k=5, accumulator=None)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10))
    outputs = knn_torch.predict(x)
    assert outputs['scores'].shape == (3, )
    assert outputs['preds'] is None
    assert outputs['p_vals'] is None
    scores = knn_torch(x)
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
    scores = knn_torch(x)
    assert scores.shape == (3,)

    knn_torch.infer_threshold(x_ref, 0.1)
    outputs = knn_torch.predict(x)
    assert torch.all(outputs['preds'] == torch.tensor([False, False, False]))

    x = torch.randn((1, 10)) * 100
    assert knn_torch(x)


def test_knn_kernel(accumulator):
    kernel = GaussianRBF(sigma=torch.tensor((0.1)))
    knn_torch = KNNTorch(k=[4, 5], kernel=kernel, accumulator=accumulator)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10))
    scores = knn_torch(x)
    assert scores.shape == (3,)

    knn_torch.infer_threshold(x_ref, 0.1)
    outputs = knn_torch.predict(x)
    assert torch.all(outputs['preds'] == torch.tensor([False, False, False]))

    x = torch.randn((1, 10)) * 100
    print(knn_torch(x))
    assert knn_torch(x)
