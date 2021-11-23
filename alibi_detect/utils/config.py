import numpy as np
import tensorflow as tf
import torch
from pydantic import BaseModel
from typing import Optional, Union, Literal, Dict, List, Callable
from alibi_detect.version import __version__
from alibi_detect.cd.tensorflow import HiddenOutput, UAE
from alibi_detect.models.tensorflow import TransformerEmbedding
from transformers import PreTrainedTokenizerBase

SUPPORTED_MODELS = Union[UAE, HiddenOutput, tf.keras.Sequential, tf.keras.Model]


class DetectorConfig(BaseModel):
    type: str
    version: str = __version__
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'


class ModelConfig(BaseModel):
    type: str = Literal['custom', 'HiddenOutput', 'UAE']
    src: str
    custom_obj: Optional[dict] = None


class EmbeddingConfig(BaseModel):
    type: str = Literal['hidden_state']  # TODO - add other types
    layers: List[int]
    src: str


class TokenizerConfig(BaseModel):
    src: str
    kwargs: Optional[dict] = {}


class PreprocessConfig(BaseModel):
    function: Union[str] = "@preprocess_drift"

    # Below kwargs are only passed if function == @preprocess_drift
    model: Union[str, ModelConfig, None] = None
    embedding: Union[str, EmbeddingConfig, None] = None
    tokenizer: Union[str, TokenizerConfig, None] = None
    device: Literal['cpu', 'cuda', 'gpu', None] = None
    preprocess_batch_fn: Optional[str] = None
    max_len: Optional[int] = None
    batch_size: Optional[int] = int(1e10)
    dtype: Optional[str] = None

    # Additional kwargs
    kwargs: dict = {}


class PreprocessConfigResolved(PreprocessConfig):
    function: Callable
    device: Optional[torch.device] = None  # Note: `device` resolved for preprocess_drift but not detectors
    model: Optional[SUPPORTED_MODELS] = None
    embedding: Optional[TransformerEmbedding] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    preprocess_batch_fn: Optional[Callable] = None
    dtype: Optional[type] = None

    class Config:
        arbitrary_types_allowed = True


class KernelConfig(BaseModel):
    kernel: str = "@GaussianRBF"

    # Below kwargs are only passed if kernel == @GaussianRBF
    sigma: Optional[List[float]] = None
    trainable: bool = False

    # Additional kwargs
    kwargs: dict = {}


class KernelConfigResolved(KernelConfig):
    kernel: Callable
    sigma: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True


class DriftDetectorConfig(DetectorConfig):
    # args/kwargs shared by all drift detectors
    x_ref: str = 'x_ref.npy'
    p_val: float = .05
    x_ref_preprocessed: bool = False
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    preprocess_fn: Union[str, PreprocessConfig, None]
    input_shape: Optional[tuple] = None  # TODO: not all offline detectors have input_shape and data_type. Should they?
    data_type: Optional[str] = None


class DriftDetectorConfigResolved(DriftDetectorConfig):
    x_ref: np.ndarray
    preprocess_fn: Union[Callable, PreprocessConfigResolved, None]

    class Config:
        arbitrary_types_allowed = True


class KSDriftConfig(DriftDetectorConfig):
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class MMDDriftConfig(DriftDetectorConfig):
    # TODO - conditional check of kernel type depending on backend
    kernel: Union[str, KernelConfig, None] = None
    sigma: Optional[List[float]] = None
    configure_kernel_from_x_ref: bool = True
    n_permutations: int = 100
    device: Literal['cpu', 'cuda', None] = None


class LSDDDriftConfig(DriftDetectorConfig):
    sigma: Optional[List[float]] = None
    n_permutations: int = 100
    n_kernel_centers: Optional[int] = None
    lambda_rd_max: float = 0.2


class KSDriftConfigResolved(DriftDetectorConfigResolved, KSDriftConfig):
    pass


class MMDDriftConfigResolved(DriftDetectorConfigResolved, MMDDriftConfig):
    kernel: Union[Callable, KernelConfigResolved, None]
    sigma: Optional[np.ndarray]


class LSDDDriftConfigResolved(DriftDetectorConfigResolved, LSDDDriftConfig):
    sigma: Optional[np.ndarray]


DETECTOR_CONFIGS = {
    'KSDrift': KSDriftConfig,
    'MMDDrift': MMDDriftConfig,
    'LSDDDrift': LSDDDriftConfig,
}


DETECTOR_CONFIGS_RESOLVED = {
    'KSDrift': KSDriftConfigResolved,
    'MMDDrift': MMDDriftConfigResolved,
    'LSDDDrift': LSDDDriftConfigResolved,
}
