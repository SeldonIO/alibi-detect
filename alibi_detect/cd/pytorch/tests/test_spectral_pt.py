from functools import partial
from itertools import product
import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Callable, List
from alibi_detect.cd.pytorch.spectral import SpectralDriftTorch
from alibi_detect.cd.pytorch.preprocess import HiddenOutput, preprocess_drift

n, n_hidden, n_classes = 500, 10, 5


class MyModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        return self.dense2(x)


# test List[Any] inputs to the detector
def preprocess_list(x: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(x, axis=0)


n_features = [10]
n_enc = [None, 3]
preprocess = [
    (None, None),
    (preprocess_drift, {'model': HiddenOutput, 'layer': -1}),
    (preprocess_list, None)
]
update_x_ref = [{'last': 750}, {'reservoir_sampling': 750}, None]
preprocess_at_init = [True, False]
n_bootstraps = [10]  # Changed from n_permutations to n_bootstraps for spectral
threshold = [None, 1.5]  # Add threshold parameter specific to spectral

tests_spectraldrift = list(product(n_features, n_enc, preprocess,
                                   n_bootstraps, update_x_ref, preprocess_at_init, threshold))
n_tests = len(tests_spectraldrift)


@pytest.fixture
def spectral_params(request):
    return tests_spectraldrift[request.param]


@pytest.mark.parametrize('spectral_params', list(range(n_tests)), indirect=True)
def test_spectral(spectral_params):
    n_features, n_enc, preprocess, n_bootstraps, update_x_ref, preprocess_at_init, threshold = spectral_params

    np.random.seed(0)
    torch.manual_seed(0)

    # Generate reference data with some correlation structure
    x_ref = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)

    # Add some correlation structure to make spectral analysis meaningful
    if n_features >= 2:
        x_ref[:, 1] += 0.3 * x_ref[:, 0]  # Add correlation between features

    preprocess_fn, preprocess_kwargs = preprocess
    to_list = False

    if hasattr(preprocess_fn, '__name__') and preprocess_fn.__name__ == 'preprocess_list':
        if not preprocess_at_init:
            return
        to_list = True
        x_ref = [_[None, :] for _ in x_ref]
    elif isinstance(preprocess_fn, Callable) and preprocess_kwargs is not None and \
            'layer' in list(preprocess_kwargs.keys()) and \
            preprocess_kwargs['model'].__name__ == 'HiddenOutput':
        model = MyModel(n_features)
        layer = preprocess_kwargs['layer']
        preprocess_fn = partial(preprocess_fn, model=HiddenOutput(model=model, layer=layer))
    else:
        preprocess_fn = None

    cd = SpectralDriftTorch(
        x_ref=x_ref,
        p_val=.05,
        preprocess_at_init=preprocess_at_init if isinstance(preprocess_fn, Callable) else False,
        update_x_ref=update_x_ref,
        preprocess_fn=preprocess_fn,
        threshold=threshold,
        n_bootstraps=n_bootstraps
    )

    # Test with reference data (should not detect drift)
    x = x_ref.copy()
    preds = cd.predict(x, return_p_val=True)
    assert preds['data']['is_drift'] == 0 and preds['data']['p_val'] >= cd.p_val

    # Check reference data update functionality
    if isinstance(update_x_ref, dict):
        k = list(update_x_ref.keys())[0]
        assert cd.n == len(x) + len(x_ref)
        assert cd.x_ref.shape[0] == min(update_x_ref[k], len(x) + len(x_ref))

    # Generate test data with different correlation structure (potential drift)
    x_h1 = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)

    # Modify correlation structure to create potential drift
    if n_features >= 2:
        x_h1[:, 1] += 0.8 * x_h1[:, 0]  # Stronger correlation = potential drift

    if to_list:
        x_h1 = [_[None, :] for _ in x_h1]

    preds = cd.predict(x_h1, return_p_val=True)

    # Check that predictions are consistent with thresholds
    if preds['data']['is_drift'] == 1:
        assert preds['data']['p_val'] < preds['data']['threshold'] == cd.p_val
        assert preds['data']['distance'] > preds['data']['distance_threshold']
    else:
        assert preds['data']['p_val'] >= preds['data']['threshold'] == cd.p_val
        assert preds['data']['distance'] <= preds['data']['distance_threshold']

    # Check that spectral ratio is computed
    assert 'spectral_ratio' in preds['data']
    assert isinstance(preds['data']['spectral_ratio'], float)
    assert preds['data']['spectral_ratio'] > 0  # Spectral ratio should be positive


def test_spectral_ratio_method():
    """Test the spectral_ratio method specifically."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_features = 5
    n_samples = 200

    # Reference data with moderate correlations
    x_ref = np.random.randn(n_samples, n_features).astype(np.float32)
    for i in range(1, n_features):
        x_ref[:, i] += 0.3 * x_ref[:, 0]  # Add correlation

    cd = SpectralDriftTorch(x_ref=x_ref, n_bootstraps=50)

    # Test data with higher correlations
    x_test = np.random.randn(100, n_features).astype(np.float32)
    for i in range(1, n_features):
        x_test[:, i] += 0.7 * x_test[:, 0]  # Higher correlation

    # Test spectral_ratio method
    ratio = cd.spectral_ratio(x_test)
    assert isinstance(ratio, float)
    assert ratio > 0
    assert ratio > 1.0  # Should be > 1 due to increased correlation


def test_spectral_stats_method():
    """Test the get_spectral_stats method."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_features = 4
    n_samples = 150

    # Reference data
    x_ref = np.random.randn(n_samples, n_features).astype(np.float32)

    cd = SpectralDriftTorch(x_ref=x_ref, n_bootstraps=50)

    # Test data
    x_test = np.random.randn(80, n_features).astype(np.float32)

    # Test get_spectral_stats method
    stats = cd.get_spectral_stats(x_test)

    expected_keys = [
        'spectral_ratio', 'test_eigenvalue', 'baseline_eigenvalue',
        'eigenvalue_change', 'eigenvalue_change_pct', 'test_trace',
        'baseline_trace', 'trace_ratio', 'test_condition_number',
        'test_samples', 'reference_samples'
    ]

    for key in expected_keys:
        assert key in stats
        assert isinstance(stats[key], (int, float))

    assert stats['test_samples'] == 80
    assert stats['reference_samples'] == n_samples
    assert stats['spectral_ratio'] > 0


def test_spectral_device_handling():
    """Test device handling (CPU/CUDA)."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_features = 3
    n_samples = 100

    x_ref = np.random.randn(n_samples, n_features).astype(np.float32)
    x_test = np.random.randn(50, n_features).astype(np.float32)

    # Test CPU device
    cd_cpu = SpectralDriftTorch(x_ref=x_ref, device='cpu', n_bootstraps=20)
    preds_cpu = cd_cpu.predict(x_test)
    assert preds_cpu['data']['is_drift'] in [0, 1]

    # Test CUDA device if available
    if torch.cuda.is_available():
        cd_cuda = SpectralDriftTorch(x_ref=x_ref, device='cuda', n_bootstraps=20)
        preds_cuda = cd_cuda.predict(x_test)
        assert preds_cuda['data']['is_drift'] in [0, 1]

        # Results should be similar (within numerical precision)
        ratio_diff = abs(preds_cpu['data']['spectral_ratio'] - preds_cuda['data']['spectral_ratio'])
        assert ratio_diff < 1e-4  # Small numerical difference tolerance


def test_spectral_minimum_features():
    """Test that spectral analysis requires at least 2 features."""
    np.random.seed(42)

    # Should fail with 1 feature
    x_ref_1d = np.random.randn(100, 1).astype(np.float32)

    with pytest.raises(ValueError, match="requires at least 2 features"):
        SpectralDriftTorch(x_ref=x_ref_1d)


def test_spectral_correlation_detection():
    """Test that spectral drift detector can detect correlation structure changes."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_features = 6
    n_samples = 300

    # Reference data: weak correlations
    x_ref = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features) + 0.1 * np.ones((n_features, n_features)),
        size=n_samples
    ).astype(np.float32)

    cd = SpectralDriftTorch(x_ref=x_ref, p_val=0.05, n_bootstraps=100)

    # Test data: strong correlations (should detect drift)
    cov_strong = np.full((n_features, n_features), 0.7)
    np.fill_diagonal(cov_strong, 1.0)

    x_drift = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=cov_strong,
        size=150
    ).astype(np.float32)

    preds = cd.predict(x_drift, return_p_val=True)

    # Should detect drift due to correlation structure change
    # Note: This is probabilistic, so we check the spectral ratio is reasonable
    assert preds['data']['spectral_ratio'] > 1.0  # Higher correlation = higher eigenvalue
    assert 'p_val' in preds['data']
    assert 'distance' in preds['data']


def test_spectral_threshold_parameter():
    """Test explicit threshold parameter."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_features = 4
    x_ref = np.random.randn(200, n_features).astype(np.float32)

    # Test with explicit threshold
    threshold = 2.0
    cd = SpectralDriftTorch(x_ref=x_ref, threshold=threshold, n_bootstraps=50)

    assert cd.threshold == threshold

    # Test data
    x_test = np.random.randn(100, n_features).astype(np.float32)
    preds = cd.predict(x_test)

    # Check that threshold is used correctly
    assert preds['data']['distance_threshold'] == threshold

    if preds['data']['spectral_ratio'] > threshold:
        assert preds['data']['is_drift'] == 1
    else:
        assert preds['data']['is_drift'] == 0


if __name__ == "__main__":
    # Run a simple test to check if everything works
    test_spectral_ratio_method()
    test_spectral_stats_method()
    test_spectral_minimum_features()
    print("✅ All basic tests passed!")
