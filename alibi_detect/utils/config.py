# mypy: ignore-errors
# TODO - conditional checks depending on backend
# TODO - defaults are currently mix of actual default and None. Doesn't matter too much as None will then be overridden
#  by detector default kwarg anyway. We could have actual defaults here for clarity, but more maintenance.
# TODO - similar for Optional[]. Many detector kwargs are not Optional[], but they are for config schema as detector
#  overrides None. What do we want to put in config schema?
# TODO - conditional backend imports
# TODO - consider validating output of get_config calls
import numpy as np
import tensorflow as tf
import torch
from pydantic import BaseModel
from typing import Optional, Union, Literal, Dict, List, Callable
from alibi_detect.version import __version__
from alibi_detect.cd.tensorflow import HiddenOutput, UAE
from alibi_detect.utils.tensorflow.data import TFDataset
from alibi_detect.models.tensorflow import TransformerEmbedding
from transformers import PreTrainedTokenizerBase

__config_spec__ = "0.1.0dev"  # TODO - remove dev once config layout confirmed

SUPPORTED_MODELS = Union[UAE, HiddenOutput, tf.keras.Sequential, tf.keras.Model]
SupportedModels = (UAE, HiddenOutput, tf.keras.Sequential, tf.keras.Model)


# Custom BaseModel so that we can set default config
class CustomBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = 'forbid'  # Forbid extra fields so that we catch misspelled fields


class DetectorConfig(CustomBaseModel):
    name: str
    version: str = __version__
    config_spec: str = __config_spec__
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    # Note: Although not all detectors have a backend, we define in base class as `backend` also determines
    #  whether tf or torch models used for preprocess_fn.


class ModelConfig(CustomBaseModel):
    type: Literal['custom', 'HiddenOutput', 'UAE'] = 'custom'
    src: str
    custom_obj: Optional[dict] = None


class EmbeddingConfig(CustomBaseModel):
    type: Literal['pooler_output', 'last_hidden_state',
                  'hidden_state', 'hidden_state_cls']  # See alibi_detect.models.tensorflow.embedding
    layers: List[int]  # TODO - add check conditional on embedding type (see docstring in above)
    src: str


class TokenizerConfig(CustomBaseModel):
    src: str
    kwargs: Optional[dict] = {}


class PreprocessConfig(CustomBaseModel):
    src: str = "@preprocess_drift"

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
    src: Callable
    device: Optional[torch.device] = None  # Note: `device` resolved for preprocess_drift but not detectors
    model: Optional[SUPPORTED_MODELS] = None
    embedding: Optional[TransformerEmbedding] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    preprocess_batch_fn: Optional[Callable] = None
    dtype: Optional[type] = None


class KernelConfig(CustomBaseModel):
    src: str = "@GaussianRBF"

    # Below kwargs are only passed if kernel == @GaussianRBF
    sigma: Optional[List[float]] = None
    trainable: bool = False

    # Additional kwargs
    kwargs: dict = {}


class KernelConfigResolved(KernelConfig):
    src: Callable
    sigma: Optional[np.ndarray] = None


class DriftDetectorConfig(DetectorConfig):
    # args/kwargs shared by all drift detectors
    x_ref: str = 'x_ref.npy'
    p_val: float = .05
    x_ref_preprocessed: bool = False
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    preprocess_fn: Union[str, PreprocessConfig, None]
    input_shape: Optional[tuple] = None
    data_type: Optional[str] = None


class DriftDetectorConfigResolved(DriftDetectorConfig):
    x_ref: np.ndarray
    preprocess_fn: Union[Callable, PreprocessConfigResolved, None]


class KSDriftConfig(DriftDetectorConfig):
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class ChiSquareDriftConfig(DriftDetectorConfig):
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Union[int, List[int]]] = None,
    n_features: Optional[int] = None


class TabularDriftConfig(DriftDetectorConfig):
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Union[int, List[int], None]] = None,
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class CVMDriftConfig(DriftDetectorConfig):
    correction: str = 'bonferroni'
    n_features: Optional[int] = None


class FETDriftConfig(DriftDetectorConfig):
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class MMDDriftConfig(DriftDetectorConfig):
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


class ClassifierDriftConfig(DriftDetectorConfig):
    model: Union[str, ModelConfig]
    preds_type: Literal['probs', 'logits'] = 'probs'
    binarize_preds: bool = False
    reg_loss_fn: Optional[str] = None
    train_size: Optional[float] = .75
    n_folds: Optional[int] = None
    retrain_from_scratch: bool = True
    seed: int = 0
    optimizer: Union[str, dict, None] = None  # dict as can pass dict to tf.keras.optimizers.deserialize
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[str] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: str = '@alibi_detect.utils.tensorflow.data.TFDataset'


class SpotTheDiffDriftConfig(ClassifierDriftConfig):
    kernel: Union[str, KernelConfig, None] = None
    n_diffs: int = 1
    initial_diffs: Optional[str] = None
    l1_reg: float = 0.01


class KSDriftConfigResolved(DriftDetectorConfigResolved, KSDriftConfig):
    pass


class ChiSquareDriftConfigResolved(DriftDetectorConfigResolved, ChiSquareDriftConfig):
    pass


class TabularDriftConfigResolved(DriftDetectorConfigResolved, TabularDriftConfig):
    pass


class CVMDriftConfigResolved(DriftDetectorConfigResolved, CVMDriftConfig):
    pass


class FETDriftConfigResolved(DriftDetectorConfigResolved, FETDriftConfig):
    pass


class MMDDriftConfigResolved(DriftDetectorConfigResolved, MMDDriftConfig):
    kernel: Union[Callable, KernelConfigResolved, None]
    sigma: Optional[np.ndarray]


class LSDDDriftConfigResolved(DriftDetectorConfigResolved, LSDDDriftConfig):
    sigma: Optional[np.ndarray]


class ClassifierDriftResolved(DriftDetectorConfigResolved, ClassifierDriftConfig):
    reg_loss_fn: Optional[Callable] = None
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    preprocess_batch_fn: Optional[Callable] = None
    dataset: Callable = TFDataset,
    model: Optional[SUPPORTED_MODELS] = None


class SpotTheDiffDriftResolved(DriftDetectorConfigResolved, SpotTheDiffDriftConfig):
    reg_loss_fn: Optional[Callable] = None
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    preprocess_batch_fn: Optional[Callable] = None
    dataset: Callable = TFDataset,
    model: Optional[SUPPORTED_MODELS] = None
    kernel: Union[Callable, KernelConfigResolved, None]
    initial_diffs: Optional[np.ndarray] = None


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
}


DETECTOR_CONFIGS_RESOLVED = {
    'KSDrift': KSDriftConfigResolved,
    'ChiSquareDrift': ChiSquareDriftConfigResolved,
    'TabularDrift': TabularDriftConfigResolved,
    'CVMDrift': CVMDriftConfigResolved,
    'FETDrift': FETDriftConfigResolved,
    'MMDDrift': MMDDriftConfigResolved,
    'LSDDDrift': LSDDDriftConfigResolved,
    'ClassifierDrift': ClassifierDriftResolved,
    'SpotTheDiffDrift': SpotTheDiffDriftResolved
}
