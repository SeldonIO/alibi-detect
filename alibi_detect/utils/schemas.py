# type: ignore[assignment]
# TODO - conditional checks depending on backend
# TODO - consider validating output of get_config calls

import numpy as np
from pydantic import BaseModel
from typing import Optional, Union, Dict, List, Callable
from alibi_detect.utils._types import Literal
from alibi_detect.version import __version__, __config_spec__
from alibi_detect.cd.tensorflow import HiddenOutput, UAE
from alibi_detect.models.tensorflow import TransformerEmbedding
from transformers import PreTrainedTokenizerBase
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow  # , has_sklearn


# Populate supported models
# SupportedModels is a tuple of the real class types, to be used in isinstance() in saving.py and loading.py.
# SUPPORTED_MODELS is a typing.Union of the same model types, to be used in type hints.
SupportedModels = [UAE, HiddenOutput]
if has_tensorflow:
    import tensorflow as tf
    SupportedModels.append(tf.keras.Model)
if has_pytorch:
    import torch
#    SupportedModels.append()  # TODO
# if has_sklearn:
#    SupportedModels.append()  # TODO
SupportedModels = tuple(SupportedModels)
SUPPORTED_MODELS = Union[SupportedModels]


# Custom BaseModel so that we can set default config
class CustomBaseModel(BaseModel):
    """
    Base pydantic model schema. Set our pydantic default settings in here.
    """
    class Config:
        arbitrary_types_allowed = True  # since we have np.ndarray's etc
        extra = 'forbid'  # Forbid extra fields so that we catch misspelled fields


class DetectorConfig(CustomBaseModel):
    """
    Base detector config schema. Only fields universal across all detectors should be defined in here.
    """
    name: str
    version: str = __version__
    config_spec: str = __config_spec__
    backend: Literal['tensorflow', 'pytorch', 'sklearn'] = 'tensorflow'
    # Note: Although not all detectors have a backend, we define in base class as `backend` also determines
    #  whether tf or torch models used for preprocess_fn.
    version_warning: bool = False


class ModelConfig(CustomBaseModel):
    """
    Unresolved schema for (ML) models. Note that the model "backend" e.g. 'tensorflow', 'pytorch', 'sklearn', is set
    by backend in DetectorConfig.
    """
    type: Literal['custom', 'HiddenOutput', 'UAE'] = 'custom'
    src: str
    custom_obj: Optional[dict] = None


class EmbeddingConfig(CustomBaseModel):
    """
    Unresolved schema for text embeddings.
    """
    type: Literal['pooler_output', 'last_hidden_state',
                  'hidden_state', 'hidden_state_cls']  # See alibi_detect.models.tensorflow.embedding
    layers: Optional[List[int]] = None  # TODO - add check conditional on embedding type (see docstring in above)
    src: str


class TokenizerConfig(CustomBaseModel):
    """
    Unresolved schema for text tokenizers.
    """
    src: str
    kwargs: Optional[dict] = {}


class PreprocessConfig(CustomBaseModel):
    """
    Unresolved schema for preprocess_fn.
    """
    src: str = "@cd.tensorflow.preprocess.preprocess_drift"

    # Below kwargs are only passed if function == @preprocess_drift
    model: Optional[Union[str, ModelConfig]] = None  # TODO - make required field when src is preprocess_drift
    embedding: Optional[Union[str, EmbeddingConfig]] = None
    tokenizer: Optional[Union[str, TokenizerConfig]] = None
    device: Optional[Literal['cpu', 'cuda']] = None
    preprocess_batch_fn: Optional[str] = None
    max_len: Optional[int] = None
    batch_size: Optional[int] = int(1e10)
    dtype: Optional[str] = None

    # Additional kwargs
    kwargs: dict = {}


class PreprocessConfigResolved(PreprocessConfig):
    """
    Resolved schema for preprocess_fn.
    """
    src: Callable
    device: Optional['torch.device'] = None  # Note: `device` resolved for preprocess_drift, but str for detectors
    model: Optional[SUPPORTED_MODELS] = None  # TODO - Not optional if src is preprocess_drift
    embedding: Optional[TransformerEmbedding] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    preprocess_batch_fn: Optional[Callable] = None
    dtype: Optional[type] = None


class KernelConfig(CustomBaseModel):
    """
    Unresolved schema for kernels.
    """
    src: str

    # Below kwargs are only passed if kernel == @GaussianRBF
    sigma: Optional[List[float]] = None
    trainable: bool = False

    # Additional kwargs
    kwargs: dict = {}


class KernelConfigResolved(KernelConfig):
    """
    Resolved schema for kernels.
    """
    src: Callable
    sigma: Optional[np.ndarray] = None


class DeepKernelConfig(CustomBaseModel):
    """
    Unresolved schema for DeepKernels (see utils.[backend].kernels.DeepKernel).
    """
    proj: Union[str, ModelConfig]
    kernel_a: Union[str, KernelConfig] = "@utils.tensorflow.kernels.GaussianRBF"
    kernel_b: Optional[Union[str, KernelConfig]] = None
    eps: Union[float, str] = 'trainable'


class DeepKernelConfigResolved(DeepKernelConfig):
    """
    Resolved schema for DeepKernels (see utils.[backend].kernels.DeepKernel).
    """
    proj: SUPPORTED_MODELS
    kernel_a: Union[Callable, KernelConfigResolved]
    kernel_b: Optional[Union[Callable, KernelConfigResolved]] = None


class DriftDetectorConfig(DetectorConfig):
    """
    Unresolved base schema for drift detectors.
    """
    # args/kwargs shared by all drift detectors
    x_ref: str
    p_val: float = .05
    x_ref_preprocessed: bool = False
    preprocess_fn: Optional[Union[str, PreprocessConfig]]
    input_shape: Optional[tuple] = None
    data_type: Optional[str] = None


class DriftDetectorConfigResolved(DriftDetectorConfig):
    """
    Resolved base schema for drift detectors.
    """
    x_ref: Union[np.ndarray, list]
    preprocess_fn: Optional[Union[Callable, PreprocessConfigResolved]]


class KSDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for KSDrift detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class ChiSquareDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for ChiSquareDrift detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Union[int, List[int]]] = None
    n_features: Optional[int] = None


class TabularDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for TabularDrift detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Optional[Union[int, List[int]]]] = None
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class CVMDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for CVMDrift detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    n_features: Optional[int] = None


class FETDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for FETDrift detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class MMDDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for MMDDrift detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    kernel: Optional[Union[str, KernelConfig]] = None
    sigma: Optional[List[float]] = None
    configure_kernel_from_x_ref: bool = True
    n_permutations: int = 100
    device: Optional[Literal['cpu', 'cuda']] = None


class LSDDDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for LSDDDrift detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    sigma: Optional[List[float]] = None
    n_permutations: int = 100
    n_kernel_centers: Optional[int] = None
    lambda_rd_max: float = 0.2
    device: Optional[Literal['cpu', 'cuda']] = None


class ClassifierDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for ClassifierDriftDrift detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    model: Union[str, ModelConfig]
    preds_type: Literal['probs', 'logits'] = 'probs'
    binarize_preds: bool = False
    reg_loss_fn: Optional[str] = None
    train_size: Optional[float] = .75
    n_folds: Optional[int] = None
    retrain_from_scratch: bool = True
    seed: int = 0
    optimizer: Optional[Union[str, dict]] = None  # dict as can pass dict to tf.keras.optimizers.deserialize
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[str] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: str = '@alibi_detect.utils.tensorflow.data.TFDataset'
    device: Optional[Literal['cpu', 'cuda']] = None


class SpotTheDiffDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for SpotTheDiffDrift detector.
    """
    binarize_preds: bool = False
    train_size: Optional[float] = .75
    n_folds: Optional[int] = None
    retrain_from_scratch: bool = True
    seed: int = 0
    optimizer: Optional[Union[str, dict]] = None  # dict as can pass dict to tf.keras.optimizers.deserialize
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[str] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: str = '@alibi_detect.utils.tensorflow.data.TFDataset'
    kernel: Optional[Union[str, KernelConfig]] = None
    n_diffs: int = 1
    initial_diffs: Optional[str] = None
    l1_reg: float = 0.01
    device: Optional[Literal['cpu', 'cuda']] = None


class LearnedKernelDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for LearnedKernelDrift detector.
    """
    kernel: Union[str, DeepKernelConfig]
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    n_permutations: int = 100
    var_reg: float = 1e-5
    reg_loss_fn: Optional[str] = None
    train_size: Optional[float] = .75
    retrain_from_scratch: bool = True
    optimizer: Optional[Union[str, dict]] = None  # dict as can pass dict to tf.keras.optimizers.deserialize
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[str] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: str = '@alibi_detect.utils.tensorflow.data.TFDataset'
    device: Optional[Literal['cpu', 'cuda']] = None


class KSDriftConfigResolved(DriftDetectorConfigResolved, KSDriftConfig):
    """
    Resolved schema for KSDrift detector.
    """
    pass


class ChiSquareDriftConfigResolved(DriftDetectorConfigResolved, ChiSquareDriftConfig):
    """
    Resolved schema for ChiSquareDrift detector.
    """
    pass


class TabularDriftConfigResolved(DriftDetectorConfigResolved, TabularDriftConfig):
    """
    Resolved schema for TabularDrift detector.
    """
    pass


class CVMDriftConfigResolved(DriftDetectorConfigResolved, CVMDriftConfig):
    pass


class FETDriftConfigResolved(DriftDetectorConfigResolved, FETDriftConfig):
    pass


class MMDDriftConfigResolved(DriftDetectorConfigResolved, MMDDriftConfig):
    """
    Resolved schema for MMDDrift detector.
    """
    kernel: Optional[Union[Callable, KernelConfigResolved]] = None
    sigma: Optional[np.ndarray] = None


class LSDDDriftConfigResolved(DriftDetectorConfigResolved, LSDDDriftConfig):
    """
    Resolved schema for LSDDDrift detector.
    """
    sigma: Optional[np.ndarray] = None


class ClassifierDriftConfigResolved(DriftDetectorConfigResolved, ClassifierDriftConfig):
    """
    Resolved schema for ClassifierDrift detector.
    """
    reg_loss_fn: Optional[Callable] = None
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    preprocess_batch_fn: Optional[Callable] = None
    dataset: Optional[Callable] = None
    model: Optional[SUPPORTED_MODELS] = None


class SpotTheDiffDriftConfigResolved(DriftDetectorConfigResolved, SpotTheDiffDriftConfig):
    """
    Resolved schema for SpotTheDiffDrift detector.
    """
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    preprocess_batch_fn: Optional[Callable] = None
    dataset: Optional[Callable] = None
    kernel: Optional[Union[Callable, KernelConfigResolved]] = None
    initial_diffs: Optional[np.ndarray] = None


class LearnedKernelDriftConfigResolved(DriftDetectorConfigResolved, LearnedKernelDriftConfig):
    """
    Resolved schema for LearnedKernelDrift detector.
    """
    kernel: Optional[Union[Callable, DeepKernelConfigResolved]] = None
    reg_loss_fn: Optional[Callable] = None
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    preprocess_batch_fn: Optional[Callable] = None
    dataset: Optional[Callable] = None


# Unresolved schema dictionary (used in alibi_detect.utils.loading)
DETECTOR_CONFIGS = {
    'KSDrift': KSDriftConfig,
    'ChiSquareDrift': ChiSquareDriftConfig,
    'TabularDrift': TabularDriftConfig,
    'CVMDrift': CVMDriftConfig,
    'FETDrift': FETDriftConfig,
    'MMDDrift': MMDDriftConfig,
    'LSDDDrift': LSDDDriftConfig,
    'ClassifierDrift': ClassifierDriftConfig,
    'SpotTheDiffDrift': SpotTheDiffDriftConfig,
    'LearnedKernelDrift': LearnedKernelDriftConfig
}


# Resolved schema dictionary (used in alibi_detect.utils.loading)
DETECTOR_CONFIGS_RESOLVED = {
    'KSDrift': KSDriftConfigResolved,
    'ChiSquareDrift': ChiSquareDriftConfigResolved,
    'TabularDrift': TabularDriftConfigResolved,
    'CVMDrift': CVMDriftConfigResolved,
    'FETDrift': FETDriftConfigResolved,
    'MMDDrift': MMDDriftConfigResolved,
    'LSDDDrift': LSDDDriftConfigResolved,
    'ClassifierDrift': ClassifierDriftConfigResolved,
    'SpotTheDiffDrift': SpotTheDiffDriftConfigResolved,
    'LearnedKernelDrift': LearnedKernelDriftConfigResolved
}
