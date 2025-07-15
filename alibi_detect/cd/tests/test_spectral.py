import numpy as np
import pytest
import torch
from alibi_detect.cd.spectral import SpectralDrift
from alibi_detect.cd.pytorch.spectral import SpectralDriftTorch

# Test data parameters
n_samples = 100
n_features = 5

@pytest.fixture
def sample_data():
    """Generate simple test data."""
    np.random.seed(42)
    x_ref = np.random.randn(n_samples, n_features).astype('float32')
    x_test = np.random.randn(80, n_features).astype('float32')
    return x_ref, x_test

def test_spectral_drift_basic_initialization(sample_data):
    """Test basic SpectralDrift initialization."""
    x_ref, _ = sample_data
    
    # Test base class
    detector = SpectralDrift(x_ref=x_ref, p_val=0.05)
    assert detector.p_val == 0.05
    assert detector.x_ref is not None
    assert hasattr(detector, 'n_features')

def test_spectral_drift_torch_initialization(sample_data):
    """Test SpectralDriftTorch initialization."""
    x_ref, _ = sample_data
    
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    
    detector = SpectralDriftTorch(
        x_ref=x_ref,
        p_val=0.05,
        n_bootstraps=50,
        threshold=None
    )
    
    assert detector.p_val == 0.05
    assert detector.n_bootstraps == 50
    assert detector.threshold is not None
    assert hasattr(detector, 'baseline_eigenvalue')

def test_spectral_drift_torch_predict(sample_data):
    """Test SpectralDriftTorch prediction."""
    x_ref, x_test = sample_data
    
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    
    detector = SpectralDriftTorch(
        x_ref=x_ref,
        p_val=0.05,
        n_bootstraps=20,  # Small for fast testing
        threshold=0.5
    )
    
    result = detector.predict(x_test, return_p_val=True)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'meta' in result
    assert 'data' in result
    assert 'is_drift' in result['data']
    assert 'spectral_ratio' in result['data']
    assert 'p_val' in result['data']

def test_spectral_drift_torch_spectral_ratio(sample_data):
    """Test spectral ratio computation."""
    x_ref, x_test = sample_data
    
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    
    detector = SpectralDriftTorch(x_ref=x_ref, n_bootstraps=20)
    ratio = detector.spectral_ratio(x_test)
    
    assert isinstance(ratio, float)
    assert ratio > 0

def test_spectral_drift_torch_stats(sample_data):
    """Test spectral statistics."""
    x_ref, x_test = sample_data
    
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    
    detector = SpectralDriftTorch(x_ref=x_ref, n_bootstraps=20)
    stats = detector.get_spectral_stats(x_test)
    
    assert isinstance(stats, dict)
    assert 'spectral_ratio' in stats
    assert 'test_eigenvalue' in stats
    assert 'baseline_eigenvalue' in stats
    assert stats['test_samples'] == x_test.shape[0]
    assert stats['reference_samples'] == x_ref.shape[0]

def test_spectral_drift_torch_wrong_dimensions(sample_data):
    """Test error handling for wrong dimensions."""
    x_ref, _ = sample_data
    
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    
    detector = SpectralDriftTorch(x_ref=x_ref, n_bootstraps=20)
    
    # Wrong number of features
    x_wrong = np.random.randn(50, n_features + 2).astype('float32')
    
    with pytest.raises(ValueError):
        detector.predict(x_wrong)

def test_spectral_drift_torch_device_handling():
    """Test device handling."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    
    x_ref = np.random.randn(50, 3).astype('float32')
    x_test = np.random.randn(30, 3).astype('float32')
    
    # CPU device
    detector_cpu = SpectralDriftTorch(x_ref=x_ref, device='cpu', n_bootstraps=10)
    result_cpu = detector_cpu.predict(x_test)
    assert isinstance(result_cpu['data']['spectral_ratio'], float)
    
    # CUDA device (if available)
    if torch.cuda.is_available():
        detector_cuda = SpectralDriftTorch(x_ref=x_ref, device='cuda', n_bootstraps=10)
        result_cuda = detector_cuda.predict(x_test)
        assert isinstance(result_cuda['data']['spectral_ratio'], float)

def test_spectral_drift_preprocess_function(sample_data):
    """Test preprocessing function."""
    x_ref, x_test = sample_data
    
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    
    def simple_preprocess(x):
        return x / np.std(x, axis=0, keepdims=True)
    
    detector = SpectralDriftTorch(
        x_ref=x_ref,
        preprocess_fn=simple_preprocess,
        preprocess_at_init=True,
        n_bootstraps=20
    )
    
    result = detector.predict(x_test)
    assert isinstance(result, dict)
    assert 'spectral_ratio' in result['data']

def test_spectral_drift_score_method(sample_data):
    """Test the score method."""
    x_ref, x_test = sample_data
    
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    
    detector = SpectralDriftTorch(x_ref=x_ref, n_bootstraps=20, threshold=0.3)
    
    p_val, spectral_ratio, threshold = detector.score(x_test)
    
    assert isinstance(p_val, float)
    assert isinstance(spectral_ratio, float)
    assert isinstance(threshold, float)
    assert 0 <= p_val <= 1
    assert spectral_ratio > 0
    assert threshold > 0

@pytest.mark.parametrize("return_p_val", [True, False])
def test_spectral_drift_return_options(sample_data, return_p_val):
    """Test different return options."""
    x_ref, x_test = sample_data
    
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    
    detector = SpectralDriftTorch(x_ref=x_ref, n_bootstraps=20)
    result = detector.predict(x_test, return_p_val=return_p_val)
    
    if return_p_val:
        assert 'p_val' in result['data']
    else:
        assert 'p_val' not in result['data']
    
    # These should always be present
    assert 'is_drift' in result['data']
    assert 'spectral_ratio' in result['data']
    assert 'distance' in result['data']