import logging
import numpy as np
from typing import Callable, Dict, Optional, Union
from alibi_detect.utils.warnings import deprecated_alias
from alibi_detect.base import DriftConfigMixin
from alibi_detect.utils._types import TorchDeviceType

logger = logging.getLogger(__name__)


class SpectralDrift(DriftConfigMixin):
    """
    Spectral eigenvalue-based drift detector for correlation structure changes.

    This detector identifies drift by analyzing changes in the eigenvalue spectrum
    of feature covariance matrices.
    """

    def __init__(self,
                 x_ref: np.ndarray,
                 backend: str = 'numpy',
                 p_val: float = .05,
                 x_ref_preprocessed: bool = False,
                 preprocess_at_init: bool = True,
                 update_x_ref: Optional[Dict] = None,
                 preprocess_fn: Optional[Callable] = None,
                 threshold: Optional[float] = None,
                 n_bootstraps: int = 100,
                 device: Optional[Union[str, TorchDeviceType]] = None,
                 input_shape: Optional[tuple] = None,
                 data_type: Optional[str] = None) -> None:

        super().__init__()

        # Store parameters
        self.x_ref = x_ref
        self.backend = backend
        self.p_val = p_val
        self.x_ref_preprocessed = x_ref_preprocessed
        self.preprocess_at_init = preprocess_at_init
        self.update_x_ref = update_x_ref
        self.input_shape = input_shape
        self.data_type = data_type
        self.threshold = threshold
        self.n_bootstraps = n_bootstraps

        # Process preprocessing
        if preprocess_fn is not None:
            if not callable(preprocess_fn):
                raise ValueError("`preprocess_fn` is not a valid Callable.")
            self._preprocess_fn = preprocess_fn
        else:
            self._preprocess_fn = None

        # Initialize detector
        self._setup_detector()

    def _setup_detector(self):
        """Set up the detector with reference data."""
        # Validate and process reference data
        x_ref = self._validate_input(self.x_ref)

        if not self.x_ref_preprocessed and self.preprocess_at_init and self._preprocess_fn:
            x_ref = self._preprocess_fn(x_ref)

        self.x_ref_processed = x_ref
        self.n_features = x_ref.shape[1]

        # Compute baseline statistics
        self._compute_baseline()

        # Set threshold
        if self.threshold is None:
            self.threshold = self._compute_threshold()

        # Set metadata
        self.meta = {
            'detector_type': 'drift',
            'data_type': self.data_type,
            'online': False,
            'backend': self.backend
        }

        logger.info(f"SpectralDrift initialized with threshold: {self.threshold:.4f}")

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        """Validate input data."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        if x.ndim != 2:
            raise ValueError(f"Input must be 2D, got shape {x.shape}")

        if x.shape[1] < 2:
            raise ValueError(f"Need at least 2 features, got {x.shape[1]}")

        # Handle bad values
        if np.any(~np.isfinite(x)):
            logger.warning("Non-finite values found, replacing with zeros")
            x = np.nan_to_num(x)

        return x.astype(np.float64)

    def _compute_baseline(self):
        """Compute baseline spectral properties."""
        x = self.x_ref_processed

        # Standardize data
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0) + 1e-8
        x_std = (x - self.mean_) / self.std_

        # Compute correlation matrix
        self.baseline_corr_ = np.corrcoef(x_std.T)

        # Regularize to ensure positive definite
        reg = 1e-6 * np.eye(self.baseline_corr_.shape[0])
        self.baseline_corr_ += reg

        # Compute eigenvalues
        eigvals = np.linalg.eigvals(self.baseline_corr_)
        eigvals = np.real(eigvals)
        eigvals = np.sort(eigvals)[::-1]  # Descending

        self.baseline_eigvals_ = eigvals
        self.baseline_spectral_norm_ = eigvals[0]

        logger.info(f"Baseline spectral norm: {self.baseline_spectral_norm_:.3f}")

    def _compute_spectral_ratio(self, x: np.ndarray) -> float:
        """Compute spectral ratio for input data."""
        # Standardize using baseline stats
        x_std = (x - self.mean_) / self.std_

        # Compute correlation matrix
        corr = np.corrcoef(x_std.T)
        reg = 1e-6 * np.eye(corr.shape[0])
        corr += reg

        # Get largest eigenvalue
        eigvals = np.linalg.eigvals(corr)
        eigvals = np.real(eigvals)
        spectral_norm = np.max(eigvals)

        # Return ratio
        return spectral_norm / self.baseline_spectral_norm_

    def _compute_threshold(self) -> float:
        """Compute detection threshold via bootstrap."""
        logger.info(f"Computing threshold with {self.n_bootstraps} bootstraps...")

        n_samples = len(self.x_ref_processed)
        ratios = []

        for i in range(self.n_bootstraps):
            # Bootstrap sample
            idx = np.random.choice(n_samples, size=n_samples//2, replace=True)
            x_boot = self.x_ref_processed[idx]

            if len(x_boot) < 10:
                continue

            try:
                ratio = self._compute_spectral_ratio(x_boot)
                if np.isfinite(ratio):
                    ratios.append(ratio)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue

        if len(ratios) < 10:
            logger.warning("Few valid bootstrap samples, using default threshold")
            return 0.2

        # Use 2-sigma rule
        ratios = np.array(ratios)
        threshold = max(2 * np.std(ratios), 0.1)

        logger.info(f"Computed threshold: {threshold:.3f}")
        return threshold

    @deprecated_alias(X='x')
    def predict(self, x: np.ndarray, return_p_val: bool = True) -> Dict:
        """Predict drift on test data."""
        # Validate input
        x_processed = self._validate_input(x)

        # Apply preprocessing if needed
        if self._preprocess_fn and not self.x_ref_preprocessed:
            x_processed = self._preprocess_fn(x_processed)

        # Check dimensions
        if x_processed.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x_processed.shape[1]}")

        if x_processed.shape[0] < 10:
            raise ValueError(f"Need at least 10 samples, got {x_processed.shape[0]}")

        # Compute spectral ratio
        spectral_ratio = self._compute_spectral_ratio(x_processed)

        # Compute distance from expected (1.0 = no drift)
        distance = abs(spectral_ratio - 1.0)

        # Make prediction
        is_drift = int(distance > self.threshold)

        # Compute p-value (approximation)
        if return_p_val:
            if is_drift:
                p_val = max(0.001, self.p_val * np.exp(-distance))
            else:
                p_val = min(0.999, 0.5 + 0.4 * np.exp(-distance))
        else:
            p_val = None

        # Build result dictionary
        result = {
            'meta': {
                'name': 'SpectralDrift',
                'detector_type': 'drift',
                'data_type': self.data_type,
                'version': '0.1.0',
                'backend': self.backend
            },
            'data': {
                'is_drift': is_drift,
                'distance': distance,
                'threshold': self.threshold,
                'spectral_ratio': float(spectral_ratio)
            }
        }

        if return_p_val:
            result['data']['p_val'] = p_val

        return result

    def score(self, x: np.ndarray) -> float:
        """Return spectral ratio score."""
        x_val = self._validate_input(x)
        return self._compute_spectral_ratio(x_val)


def test_spectral_drift_compatibility():
    """Test SpectralDrift compatibility."""
    print("Testing SpectralDrift compatibility...")

    np.random.seed(42)

    # Generate reference data
    x_ref = np.random.randn(500, 5)

    # Generate test data with different correlation structure
    cov = np.full((5, 5), 0.7)
    np.fill_diagonal(cov, 1.0)
    x_test = np.random.multivariate_normal(np.zeros(5), cov, 200)

    # Test detector
    try:
        detector = SpectralDrift(x_ref, p_val=0.05, n_bootstraps=50)
        result = detector.predict(x_test)

        print("✅ SpectralDrift test successful")
        print(f"Drift detected: {result['data']['is_drift']}")
        print(f"Spectral ratio: {result['data']['spectral_ratio']:.3f}")
        print(f"Distance: {result['data']['distance']:.3f}")

        return True
    except Exception as e:
        print(f"❌ SpectralDrift test failed: {e}")
        return False


if __name__ == "__main__":
    test_spectral_drift_compatibility()
