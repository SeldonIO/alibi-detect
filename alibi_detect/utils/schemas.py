"""
Pydantic models used by :func:`~alibi_detect.utils.validate.validate_config` to validate configuration dictionaries.
There are pydantic models for *unresolved* and *resolved* config dictionaries:

- *Unresolved* config dictionaries are dictionaries where any artefacts specified within it have not yet been resolved.
  The artefacts are still string references to local filepaths or registries (e.g. `x_ref = 'x_ref.npy'`).

- *Resolved* config dictionaries are dictionaries containing resolved artefacts. The artefacts are expected to be
  runtime artefacts. For example, `x_ref` should have been resolved into an `np.ndarray`.

.. note::
    For detector pydantic models, the fields match the corresponding detector's args/kwargs. Refer to the
    detector's api docs for a full description of each arg/kwarg.
"""

# TODO - conditional checks depending on backend etc
# TODO - consider validating output of get_config calls
import numpy as np
from pydantic import BaseModel
from typing import Optional, Union, Dict, List, Callable, Any
from alibi_detect.utils._types import Literal
from alibi_detect.version import __version__, __config_spec__
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.utils.frameworks import has_tensorflow

SupportedModels = []
if has_tensorflow:
    import tensorflow as tf
    from alibi_detect.cd.tensorflow import UAE, HiddenOutput
    SupportedModels += [tf.keras.Model, UAE, HiddenOutput]
# if has_pytorch:
#    import torch
#    SupportedModels.append()  # TODO
# if has_sklearn:
#    import sklearn
#    SupportedModels.append()  # TODO
# SupportedModels is a tuple of possible models (conditional on installed deps). This is used in isinstance() etc.
SupportedModels = tuple(SupportedModels)  # type: ignore[assignment]
# SupportedModels_py is a typing Union only for use with pydantic. NOT to be used with mypy (as not static)
SupportedModels_py = Union[SupportedModels]  # type: ignore[valid-type]


# Custom BaseModel so that we can set default config
class CustomBaseModel(BaseModel):
    """
    Base pydantic model schema. The default pydantic settings are set here.
    """
    class Config:
        arbitrary_types_allowed = True  # since we have np.ndarray's etc
        extra = 'forbid'  # Forbid extra fields so that we catch misspelled fields


class DetectorConfig(CustomBaseModel):
    """
    Base detector config schema. Only fields universal across all detectors are defined here.
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
    by `backend` in :class:`DetectorConfig`.
    """
    src: str
    custom_objects: Optional[dict] = None
    layer: Optional[int] = None


class EmbeddingConfig(CustomBaseModel):
    """
    Unresolved schema for text embedding models.

    Currently, only HuggingFace transformer models are supported, via
    :class:`~alibi_detect.models.tensorflow.embedding.TransformerEmbedding`.
    """
    type: Literal['pooler_output', 'last_hidden_state',
                  'hidden_state', 'hidden_state_cls']  # See alibi_detect.models.tensorflow.embedding
    layers: Optional[List[int]] = None  # TODO - add check conditional on embedding type (see docstring in above)
    src: str


class TokenizerConfig(CustomBaseModel):
    """
    Unresolved schema for text tokenizers.

    Currently, only HuggingFace tokenizer models are supported, with the `src` field passed to
    :meth:`transformers.AutoTokenizer.from_pretrained`.
    """
    src: str
    kwargs: Optional[dict] = {}


class PreprocessConfig(CustomBaseModel):
    """
    Unresolved schema for drift detector preprocessors, to be passed to the `preprocess_fn` kwarg.
    """
    src: str = "@cd.tensorflow.preprocess.preprocess_drift"

    # Below kwargs are only passed if src == @preprocess_drift
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


class PreprocessConfigResolved(CustomBaseModel):
    """
    Resolved schema for drift detector preprocessors, to be passed to a detector's `preprocess_fn` kwarg.
    """
    src: Callable

    # Below kwargs are only passed if src == @preprocess_drift
    model: Optional[SupportedModels_py] = None  # TODO - Not optional if src is preprocess_drift
    embedding: Optional[TransformerEmbedding] = None
    tokenizer: Optional[Callable] = None  # TODO - PreTrainedTokenizerBase currently causes docs issues
    device: Optional[Any] = None  # TODO: Set as none for now. Think about how to handle ForwardRef when torch missing
    preprocess_batch_fn: Optional[Callable] = None
    max_len: Optional[int] = None
    batch_size: Optional[int] = int(1e10)
    dtype: Optional[type] = None

    # Additional kwargs
    kwargs: dict = {}


class KernelConfig(CustomBaseModel):
    """
    Unresolved schema for kernels, to be passed to a detector's `kernel` kwarg.
    """
    src: str

    # Below kwargs are only passed if kernel == @GaussianRBF
    sigma: Optional[List[float]] = None
    trainable: bool = False

    # Additional kwargs
    kwargs: dict = {}


class KernelConfigResolved(CustomBaseModel):
    """
    Resolved schema for kernels, to be passed to a detector's `kernel` kwarg.
    """
    src: Callable

    # Below kwargs are only passed if kernel == @GaussianRBF
    sigma: Optional[np.ndarray] = None
    trainable: bool = False

    # Additional kwargs
    kwargs: dict = {}


class DeepKernelConfig(CustomBaseModel):
    """
    Unresolved schema for :class:`~alibi_detect.utils.tensorflow.kernels.DeepKernel`'s.
    """
    proj: Union[str, ModelConfig]
    kernel_a: Union[str, KernelConfig] = "@utils.tensorflow.kernels.GaussianRBF"
    kernel_b: Optional[Union[str, KernelConfig]] = None
    eps: Union[float, str] = 'trainable'


class DeepKernelConfigResolved(CustomBaseModel):
    """
    Resolved schema for :class:`~alibi_detect.utils.tensorflow.kernels.DeepKernel`'s.
    """
    proj: SupportedModels_py
    kernel_a: Union[Callable, KernelConfigResolved]
    kernel_b: Optional[Union[Callable, KernelConfigResolved]] = None
    eps: Union[float, str] = 'trainable'


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


class DriftDetectorConfigResolved(DetectorConfig):
    """
    Resolved base schema for drift detectors.
    """
    # args/kwargs shared by all drift detectors
    x_ref: Union[np.ndarray, list]
    p_val: float = .05
    x_ref_preprocessed: bool = False
    preprocess_fn: Optional[Union[Callable, PreprocessConfigResolved]]
    input_shape: Optional[tuple] = None
    data_type: Optional[str] = None


class KSDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the :class:`~alibi_detect.cd.KSDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class KSDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the :class:`~alibi_detect.cd.KSDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class ChiSquareDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the :class:`~alibi_detect.cd.ChiSquareDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Union[int, List[int]]] = None
    n_features: Optional[int] = None


class ChiSquareDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the :class:`~alibi_detect.cd.ChiSquareDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Union[int, List[int]]] = None
    n_features: Optional[int] = None


class TabularDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the :class:`~alibi_detect.cd.TabularDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Optional[Union[int, List[int]]]] = None
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class TabularDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the :class:`~alibi_detect.cd.TabularDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Optional[Union[int, List[int]]]] = None
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class CVMDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the :class:`~alibi_detect.cd.CVMDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    n_features: Optional[int] = None


class CVMDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the :class:`~alibi_detect.cd.CVMDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    n_features: Optional[int] = None


class FETDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the :class:`~alibi_detect.cd.FETDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class FETDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the :class:`~alibi_detect.cd.FETDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class MMDDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the :class:`~alibi_detect.cd.MMDDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    kernel: Optional[Union[str, KernelConfig]] = None
    sigma: Optional[List[float]] = None
    configure_kernel_from_x_ref: bool = True
    n_permutations: int = 100
    device: Optional[Literal['cpu', 'cuda']] = None


class MMDDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the :class:`~alibi_detect.cd.MMDDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    kernel: Optional[Union[Callable, KernelConfigResolved]] = None
    sigma: Optional[np.ndarray] = None
    configure_kernel_from_x_ref: bool = True
    n_permutations: int = 100
    device: Optional[Literal['cpu', 'cuda']] = None


class LSDDDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the :class:`~alibi_detect.cd.LSDDDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    sigma: Optional[List[float]] = None
    n_permutations: int = 100
    n_kernel_centers: Optional[int] = None
    lambda_rd_max: float = 0.2
    device: Optional[Literal['cpu', 'cuda']] = None


class LSDDDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the :class:`~alibi_detect.cd.LSDDDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    sigma: Optional[np.ndarray] = None
    n_permutations: int = 100
    n_kernel_centers: Optional[int] = None
    lambda_rd_max: float = 0.2
    device: Optional[Literal['cpu', 'cuda']] = None


class ClassifierDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the :class:`~alibi_detect.cd.ClassifierDrift` detector.
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


class ClassifierDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the :class:`~alibi_detect.cd.ClassifierDrift` detector.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    model: Optional[SupportedModels_py] = None
    preds_type: Literal['probs', 'logits'] = 'probs'
    binarize_preds: bool = False
    reg_loss_fn: Optional[Callable] = None
    train_size: Optional[float] = .75
    n_folds: Optional[int] = None
    retrain_from_scratch: bool = True
    seed: int = 0
    optimizer: Optional['tf.keras.optimizers.Optimizer'] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[Callable] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: Optional[Callable] = None
    device: Optional[Literal['cpu', 'cuda']] = None


class SpotTheDiffDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the :class:`~alibi_detect.cd.SpotTheDiffDrift` detector.
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


class SpotTheDiffDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the :class:`~alibi_detect.cd.SpotTheDiffDrift` detector.
    """
    binarize_preds: bool = False
    train_size: Optional[float] = .75
    n_folds: Optional[int] = None
    retrain_from_scratch: bool = True
    seed: int = 0
    optimizer: Optional['tf.keras.optimizers.Optimizer'] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[Callable] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: Optional[Callable] = None
    kernel: Optional[Union[Callable, KernelConfigResolved]] = None
    n_diffs: int = 1
    initial_diffs: Optional[np.ndarray] = None
    l1_reg: float = 0.01
    device: Optional[Literal['cpu', 'cuda']] = None


class LearnedKernelDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the :class:`~alibi_detect.cd.LearnedKernelDrift` detector.
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


class LearnedKernelDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the :class:`~alibi_detect.cd.LearnedKernelDrift` detector.
    """
    kernel: Optional[Union[Callable, DeepKernelConfigResolved]] = None
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    n_permutations: int = 100
    var_reg: float = 1e-5
    reg_loss_fn: Optional[Callable] = None
    train_size: Optional[float] = .75
    retrain_from_scratch: bool = True
    optimizer: Optional['tf.keras.optimizers.Optimizer'] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[Callable] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: Optional[Callable] = None
    device: Optional[Literal['cpu', 'cuda']] = None


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
