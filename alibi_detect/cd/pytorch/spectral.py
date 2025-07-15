import logging
import numpy as np
import torch
from typing import Callable, Dict, Optional, Tuple, Union, List, Any
from contextlib import contextmanager
from alibi_detect.cd.base import BaseSpectralDrift
from alibi_detect.utils.pytorch import get_device
from alibi_detect.utils.warnings import deprecated_alias
from alibi_detect.utils.frameworks import Framework
from alibi_detect.utils._types import TorchDeviceType

logger = logging.getLogger(__name__)


class SpectralDriftTorch(BaseSpectralDrift):
    @deprecated_alias(preprocess_x_ref='preprocess_at_init')
    def __init__(
            self,
            x_ref: Union[np.ndarray, List[Any]],
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            threshold: Optional[float] = None,
            n_bootstraps: int = 1000,
            device: TorchDeviceType = None,
            input_shape: Optional[Tuple[int, ...]] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Spectral eigenvalue-based data drift detector using PyTorch backend.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_fn=preprocess_fn,
            threshold=threshold,
            n_bootstraps=n_bootstraps
        )

        # Store additional parameters
        self.x_ref_preprocessed = x_ref_preprocessed
        self.preprocess_at_init = preprocess_at_init
        self.update_x_ref = update_x_ref
        self.input_shape = input_shape
        self.data_type = data_type

        # Add reference update support
        self.n = len(x_ref)

        # Set backend metadata
        if hasattr(self, 'meta'):
            self.meta.update({'backend': Framework.PYTORCH.value})

        # Set device
        self.device = get_device(device)

        # Process reference data
        if self.preprocess_fn is not None and self.preprocess_at_init and not self.x_ref_preprocessed:
            self.x_ref = self.preprocess_fn(x_ref)

        # Validate and convert reference data
        self._validate_input(self.x_ref)
        x_ref_tensor = self._to_tensor(self.x_ref)
        
        # Compute baseline spectral properties
        self._compute_baseline_spectrum(x_ref_tensor)

        # Infer threshold if not provided
        if self.threshold is None:
            self.threshold = self._infer_threshold(x_ref_tensor)

    def _validate_input(self, x: Union[np.ndarray, List[Any]]) -> None:
        """Validate input data dimensions and type."""
        if isinstance(x, list):
            x = np.array(x)
        
        if x.ndim != 2:
            raise ValueError(f"Input must be 2D, got shape {x.shape}")
        
        if x.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples, got {x.shape[0]}")
        
        if x.shape[1] < 2:
            raise ValueError(f"Need at least 2 features for spectral analysis, got {x.shape[1]}")

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor, List[Any]]) -> torch.Tensor:
        """Convert input to PyTorch tensor."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device).float()
        elif isinstance(x, list):
            return torch.tensor(x, device=self.device, dtype=torch.float32)
        else:  # numpy array
            return torch.from_numpy(x).to(self.device).float()

    @contextmanager
    def _device_context(self):
        """Context manager for device operations."""
        try:
            yield
        finally:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def _compute_baseline_spectrum(self, x_ref: torch.Tensor) -> None:
        """Compute baseline covariance matrix and eigenvalue spectrum using PyTorch."""
        with self._device_context():
            # Center the data
            x_centered = x_ref - torch.mean(x_ref, dim=0, keepdim=True)

            # Compute covariance matrix
            n_samples = x_ref.shape[0]
            self.baseline_cov = torch.mm(x_centered.t(), x_centered) / (n_samples - 1)

            # Compute eigenvalues using PyTorch
            eigenvals = torch.linalg.eigvals(self.baseline_cov)
            eigenvals = torch.real(eigenvals)  # Take real part
            eigenvals = torch.sort(eigenvals, descending=True)[0]  # Sort descending

            self.baseline_eigenvalues = eigenvals
            self.baseline_eigenvalue = eigenvals[0]  # Largest eigenvalue

            # Store additional baseline statistics
            self.baseline_trace = torch.sum(eigenvals)
            self.baseline_det = torch.prod(eigenvals[eigenvals > 1e-10])
            self.baseline_condition_number = eigenvals[0] / eigenvals[-1] if eigenvals[-1] > 1e-10 else float('inf')

    def _compute_test_spectrum(self, x_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute test data eigenvalue spectrum and spectral ratio using PyTorch."""
        with self._device_context():
            # Center the test data
            x_centered = x_test - torch.mean(x_test, dim=0, keepdim=True)

            # Compute test covariance matrix
            n_samples = x_test.shape[0]
            test_cov = torch.mm(x_centered.t(), x_centered) / (n_samples - 1)

            # Compute eigenvalues
            eigenvals = torch.linalg.eigvals(test_cov)
            eigenvals = torch.real(eigenvals)
            eigenvals = torch.sort(eigenvals, descending=True)[0]

            test_eigenvalue = eigenvals[0]
            spectral_ratio = test_eigenvalue / self.baseline_eigenvalue

            return spectral_ratio, test_eigenvalue, eigenvals

    def _infer_threshold(self, x_ref: torch.Tensor) -> float:
        """Infer threshold using bootstrap method with PyTorch tensors."""
        logger.info(f"Inferring threshold using {self.n_bootstraps} bootstrap samples...")

        n_samples = x_ref.shape[0]
        bootstrap_ratios = []

        for _ in range(self.n_bootstraps):
            # Bootstrap sample from reference distribution
            indices = torch.randint(0, n_samples, (max(n_samples // 2, 50),), device=self.device)
            x_bootstrap = x_ref[indices]

            # Compute spectral ratio
            ratio, _, _ = self._compute_test_spectrum(x_bootstrap)
            bootstrap_ratios.append(ratio.cpu().item())

        # Store bootstrap ratios for p-value computation
        self.bootstrap_ratios = bootstrap_ratios
        
        # Set threshold at (1-p_val) quantile
        threshold = float(np.quantile(bootstrap_ratios, 1 - self.p_val))

        logger.info(f"Inferred threshold: {threshold:.4f}")
        return threshold

    def _compute_p_value(self, spectral_ratio: float) -> float:
        """Compute p-value using bootstrap distribution."""
        if hasattr(self, 'bootstrap_ratios'):
            return float(np.mean(np.array(self.bootstrap_ratios) >= spectral_ratio))
        else:
            # Fallback to threshold-based
            return 0.01 if spectral_ratio > self.threshold else 0.5

    def score(self, x: Union[np.ndarray, List[Any]]) -> Tuple[float, float, float]:
        """Compute the spectral drift score."""
        self._validate_input(x)
        
        x_ref, x = self.preprocess(x)
        x_ref_tensor = self._to_tensor(x_ref)
        x_tensor = self._to_tensor(x)

        # Validate feature dimensions
        if x_tensor.shape[1] != x_ref_tensor.shape[1]:
            raise ValueError(f"Test data has {x_tensor.shape[1]} features, expected {x_ref_tensor.shape[1]}")

        # Compute spectral ratio
        spectral_ratio, test_eigenvalue, _ = self._compute_test_spectrum(x_tensor)

        # Compute p-value
        p_val = self._compute_p_value(spectral_ratio.cpu().item())

        return p_val, spectral_ratio.cpu().item(), self.threshold

    def predict(self, x: Union[np.ndarray, List[Any]], return_p_val: bool = True, 
               return_distance: bool = True) -> Dict[str, Any]:
        """Predict whether a batch of data has drifted from the reference data."""
        # Compute drift scores
        p_val, spectral_ratio, distance_threshold = self.score(x)
        drift_pred = int(p_val < self.p_val)

        # Handle reference data updates (simplified version)
        if isinstance(self.update_x_ref, dict):
            if isinstance(x, list):
                self.n += len(x)
            else:  # numpy array
                self.n += x.shape[0]

        # Prepare return data
        data: Dict[str, Union[int, float]] = {
            'is_drift': drift_pred,
            'distance': spectral_ratio,
            'threshold': self.p_val,
            'distance_threshold': distance_threshold,
            'spectral_ratio': spectral_ratio
        }

        if return_p_val:
            data['p_val'] = p_val

        meta: Dict[str, str] = {
            'name': 'SpectralDrift',
            'detector_type': 'drift',
            'data_type': self.data_type or 'tabular',
            'backend': 'pytorch'
        }

        return {'meta': meta, 'data': data}

    def spectral_ratio(self, x: Union[np.ndarray, List[Any]]) -> float:
        """Compute the spectral ratio between test data and reference data."""
        self._validate_input(x)
        
        x_ref, x = self.preprocess(x)
        x_ref_tensor = self._to_tensor(x_ref)
        x_tensor = self._to_tensor(x)

        spectral_ratio, _, _ = self._compute_test_spectrum(x_tensor)

        return spectral_ratio.cpu().item()

    def get_spectral_stats(self, x: Union[np.ndarray, List[Any]]) -> Dict[str, float]:
        """Get detailed spectral statistics for analysis."""
        self._validate_input(x)
        
        x_ref, x = self.preprocess(x)
        x_ref_tensor = self._to_tensor(x_ref)
        x_tensor = self._to_tensor(x)

        spectral_ratio, test_eigenvalue, test_eigenvalues = self._compute_test_spectrum(x_tensor)

        # Additional spectral statistics
        test_trace = torch.sum(test_eigenvalues)
        test_condition_number = test_eigenvalues[0] / test_eigenvalues[-1] if test_eigenvalues[-1] > 1e-10 else float('inf')

        # Ratios and changes
        trace_ratio = test_trace / self.baseline_trace
        eigenvalue_change = test_eigenvalue - self.baseline_eigenvalue
        eigenvalue_change_pct = (eigenvalue_change / self.baseline_eigenvalue) * 100

        # Convert all to CPU and numpy
        def to_cpu_item(value: Union[torch.Tensor, float]) -> float:
            """Convert tensor to CPU float or return float as-is."""
            if isinstance(value, torch.Tensor):
                return value.cpu().item()
            else:
                return float(value)

        return {
            'spectral_ratio': to_cpu_item(spectral_ratio),
            'test_eigenvalue': to_cpu_item(test_eigenvalue),
            'baseline_eigenvalue': to_cpu_item(self.baseline_eigenvalue),
            'eigenvalue_change': to_cpu_item(eigenvalue_change),
            'eigenvalue_change_pct': to_cpu_item(eigenvalue_change_pct),
            'test_trace': to_cpu_item(test_trace),
            'baseline_trace': to_cpu_item(self.baseline_trace),
            'trace_ratio': to_cpu_item(trace_ratio),
            'test_condition_number': to_cpu_item(test_condition_number),
            'baseline_condition_number': to_cpu_item(self.baseline_condition_number),
            'test_samples': x_tensor.shape[0],
            'reference_samples': x_ref_tensor.shape[0]
        }