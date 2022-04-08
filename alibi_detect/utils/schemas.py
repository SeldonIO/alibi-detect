"""
Pydantic models used by :func:`~alibi_detect.utils.validate.validate_config` to validate configuration dictionaries.
The `resolved` kwarg of :func:`~alibi_detect.utils.validate.validate_config` determines whether the *unresolved* or
*resolved* pydantic models are used:

- The *unresolved* models expect any artefacts specified within it to not yet have been resolved.
  The artefacts are still string references to local filepaths or registries (e.g. `x_ref = 'x_ref.npy'`).

- The *resolved* models expect all artefacts to be have been resolved into runtime objects. For example, `x_ref`
  should have been resolved into an `np.ndarray`.

.. note::
    For detector pydantic models, the fields match the corresponding detector's args/kwargs. Refer to the
    detector's api docs for a full description of each arg/kwarg.
"""

# TODO - conditional checks depending on backend etc
# TODO - consider validating output of get_config calls
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, List, Callable, Any
from alibi_detect.utils._types import Literal
from alibi_detect.version import __version__, __config_spec__
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.utils.frameworks import has_tensorflow

SupportedModels_li = []
if has_tensorflow:
    import tensorflow as tf
    from alibi_detect.cd.tensorflow import UAE, HiddenOutput
    SupportedModels_li += [tf.keras.Model, UAE, HiddenOutput]
# if has_pytorch:
#    import torch
#    SupportedModels_li.append()  # TODO
# if has_sklearn:
#    import sklearn
#    SupportedModels_li.append()  # TODO
# SupportedModels is a tuple of possible models (conditional on installed deps). This is used in isinstance() etc.
SupportedModels = tuple(SupportedModels_li)
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


class MetaData(CustomBaseModel):
    version: str = __version__
    config_spec: str = __config_spec__
    version_warning: bool = False


class DetectorConfig(CustomBaseModel):
    """
    Base detector config schema. Only fields universal across all detectors are defined here.
    """
    name: str = Field(..., description='Name of the detector e.g. `MMDDrift`.')
    backend: Literal['tensorflow', 'pytorch', 'sklearn'] = Field('tensorflow', description='The detector backend.')
    # Note: Although not all detectors have a backend, we define in base class as `backend` also determines
    #  whether tf or torch models used for preprocess_fn.
    meta: Optional[MetaData] = Field(None, description='Config metadata. Should not be edited.')


class ModelConfig(CustomBaseModel):
    """
    Unresolved schema for (ML) models. Note that the model "backend" e.g. 'tensorflow', 'pytorch', 'sklearn', is set
    by `backend` in :class:`DetectorConfig`.
    """
    src: str = Field(..., description='Filepath to directory storing the model (relative to the `config.toml` '
                                      'file, or absolute).')
    custom_objects: Optional[dict] \
        = Field(None, description='Dictionary of custom objects. Passed to the tensorflow '
                                  '`load_model <https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model>`_ '  # noqa: E501
                                  'function. This can be used to pass custom registered functions and classes to '
                                  'a model.')
    layer: Optional[int] \
        = Field(None, description='Optional index of hidden layer to extract. If not `None`, a '
                                  ':class:`~alibi_detect.cd.tensorflow.preprocess.HiddenOutput` model is returned.')


class EmbeddingConfig(CustomBaseModel):
    """
    Unresolved schema for text embedding models. Currently, only HuggingFace transformer models are supported.
    """
    type: Literal['pooler_output', 'last_hidden_state', 'hidden_state', 'hidden_state_cls'] \
        = Field(..., description='The type of embedding to be loaded. See `embedding_type` in '
                                 ':class:`~alibi_detect.models.tensorflow.embedding.TransformerEmbedding`.')
    layers: Optional[List[int]] \
        = Field(None, description='List specifying the hidden layers to be used to extract the embedding. ')
    # TODO - add check conditional on embedding type (see docstring in above)
    src: str \
        = Field(..., description='Model name e.g. `"bert-base-cased"`, or a filepath to directory storing the model to '
                                 'extract embeddings from (relative to the `config.toml` file, or absolute).')


class TokenizerConfig(CustomBaseModel):
    """
    Unresolved schema for text tokenizers. Currently, only HuggingFace tokenizer models are supported.
    """
    src: str \
        = Field(..., description='Model name e.g. `"bert-base-cased"`, or a filepath to directory storing the '
                                 'tokenizer model (relative to the `config.toml` file, or absolute). Passed to'
                                 'passed to :meth:`transformers.AutoTokenizer.from_pretrained`.')
    kwargs: Optional[dict] \
        = Field({}, description='Dictionary of keyword arguments to pass to '
                                ':meth:`transformers.AutoTokenizer.from_pretrained`.')


class PreprocessConfig(CustomBaseModel):
    """
    Unresolved schema for drift detector preprocess functions, to be passed to a detector's `preprocess_fn` kwarg.
    Once loaded, the function is wrapped in a :func:`~functools.partial`, to be evaluated within the detector.

    If `src` specifies a generic Python function, the dictionary specified by `kwargs` is passed to it. Otherwise,
    if `src` specifies :func:`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift`
    (`src='@cd.tensorflow.preprocess.preprocess_drift'`), all fields (except `kwargs`) are passed to it.
    """
    src: str \
        = Field("'@cd.tensorflow.preprocess.preprocess_drift'",
                description='The preprocessing function. A string referencing a filepath to a serialized function in '
                            '`.dill` format, or an object registry reference.')

    # Below kwargs are only passed if src == @preprocess_drift
    model: Optional[Union[str, ModelConfig]] \
        = Field(None, description=' Model used for preprocessing. Either an object registry reference, '
                                  'or a :class:`~alibi_detect.utils.schemas.ModelConfig`.')
    # TODO - make model required field when src is preprocess_drift
    embedding: Optional[Union[str, EmbeddingConfig]] \
        = Field(None, description='A text embedding model. Either a string referencing a HuggingFace transformer model'
                                  'name, an object registry reference, or a '
                                  ':class:`~alibi_detect.utils.schemas.EmbeddingConfig`. If `model=None`, the '
                                  '`embedding` is passed to '
                                  ':func:`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift` as `model`. '
                                  'Otherwise, the `model` is chained to the output of the `embedding` as an additional '
                                  'preprocessing step.')
    tokenizer: Optional[Union[str, TokenizerConfig]] \
        = Field(None, description='Optional tokenizer for text drift. Either a string referencing a HuggingFace '
                                  'tokenizer model name, or a :class:`~alibi_detect.utils.schemas.TokenizerConfig`.')
    device: Optional[Literal['cpu', 'cuda']] \
        = Field(None, description="Device type used. The default `None` tries to use the GPU and falls back on CPU if "
                                  "needed. Only relevant if `src='@cd.torch.preprocess.preprocess_drift'`.")
    preprocess_batch_fn: Optional[str] \
        = Field(None, description='Optional batch preprocessing function. For example to convert a list of objects to '
                                  'a batch which can be processed by the `model`.')
    max_len: Optional[int] = Field(None, description='Optional max token length for text drift.')
    batch_size: Optional[int] = Field(int(1e10), description='Batch size used during prediction.')
    dtype: Optional[str] = Field(None, description="Model output type, e.g. `'np.float32'` or `'torch.float32'`.")

    # Additional kwargs
    kwargs: dict = Field({}, description='Dictionary of keyword arguments to be passed to the function specified by '
                                         '`src`. Only used if `src` specifies a generic Python function.')


class PreprocessConfigResolved(CustomBaseModel):
    """
    Resolved schema for drift detector preprocess functions, to be passed to a detector's `preprocess_fn` kwarg.
    Once loaded, the function is wrapped in a :func:`~functools.partial`, to be evaluated within the detector.

    If `src` is a generic Python function, the dictionary specified by `kwargs` is passed to it. Otherwise,
    if `src` is a :func:`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift` function, all fields
    (except `kwargs`) are passed to it.
    """
    src: Callable \
        = Field(..., description='The preprocessing function.')

    # Below kwargs are only passed if src == @preprocess_drift
    model: Optional[SupportedModels_py] \
        = Field(None, description='Model used for preprocessing.')
    embedding: Optional[TransformerEmbedding] \
        = Field(None, description='A text embedding model. If `model=None`, the `embedding` is passed to '
                                  ':func:`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift` as `model`. '
                                  'Otherwise, the `model` is chained to the output of the `embedding` as an additional '
                                  'preprocessing step.')  # TODO - Not optional if src is preprocess_drift
    tokenizer: Optional[Callable] = Field(None, description='Optional tokenizer for text drift.')
    # TODO - typing as PreTrainedTokenizerBase currently causes docs issues, so relaxing to Callable for now
    device: Optional[Any] \
        = Field(None, description='Device type used. The default `None` tries to use the GPU and falls back on CPU if '
                                  'needed. Only relevant if function is '
                                  ':func:`~alibi_detect.cd.pytorch.preprocess.preprocess_drift`.')
    # TODO: Set as Any and None for now. Think about how to handle ForwardRef when torch missing
    preprocess_batch_fn: Optional[Callable] \
        = Field(None, description='Optional batch preprocessing function. For example to convert a list of objects to '
                                  'a batch which can be processed by the `model`.')
    max_len: Optional[int] = Field(None, description='Optional max token length for text drift.')
    batch_size: Optional[int] = Field(int(1e10), description='Batch size used during prediction.')
    dtype: Optional[str] = Field(None, description="Model output type, e.g. `'np.float32'` or `'torch.float32'`.")

    # Additional kwargs
    kwargs: dict = Field({}, description='Dictionary of keyword arguments to be passed to the function specified by '
                                         '`src`. Only used if `src` specifies a generic Python function.')


class KernelConfig(CustomBaseModel):
    """
    Unresolved schema for kernels, to be passed to a detector's `kernel` kwarg.

    If `src` specifies a :class:`~alibi_detect.utils.tensorflow.GaussianRBF` kernel, the `sigma` and `trainable` fields
    are passed to it. Otherwise, the `kwargs` field is passed.
    """
    src: str = Field(..., description='A string referencing a filepath to a serialized kernel in `.dill` format, or '
                                      'an object registry reference.')

    # Below kwargs are only passed if kernel == @GaussianRBF
    sigma: Optional[List[float]] \
        = Field(None, description='Bandwidth used for the kernel. Needn’t be specified if being inferred or trained. '
                                  'Can pass multiple values to eval kernel with and then average.')
    trainable: bool = Field(False, description='Whether or not to track gradients w.r.t. sigma to allow it to be '
                                               'trained.')

    # Additional kwargs
    kwargs: dict = Field({}, description='Dictionary of keyword arguments to pass to the kernel.')


class KernelConfigResolved(CustomBaseModel):
    """
    Resolved schema for kernels, to be passed to a detector's `kernel` kwarg.

    If `src` is a :class:`~alibi_detect.utils.tensorflow.GaussianRBF` kernel, the `sigma` and `trainable` fields
    are passed to it. Otherwise, the `kwargs` field is passed.
    """
    src: Callable = Field(..., description='The kernel.')

    # Below kwargs are only passed if kernel == @GaussianRBF
    sigma: Optional[np.ndarray] \
        = Field(None, description='Bandwidth used for the kernel. Needn’t be specified if being inferred or trained. '
                                  'Can pass multiple values to eval kernel with and then average.')

    trainable: bool = Field(False, description='Whether or not to track gradients w.r.t. sigma to allow it to be '
                                               'trained.')

    # Additional kwargs
    kwargs: dict = Field({}, description='Dictionary of keyword arguments to pass to the kernel.')


class DeepKernelConfig(CustomBaseModel):
    """
    Unresolved schema for :class:`~alibi_detect.utils.tensorflow.kernels.DeepKernel`'s.
    """
    proj: Union[str, ModelConfig] \
        = Field(..., description='The projection to be applied to the inputs before applying `kernel_a`. This should '
                                 'be a Tensorflow or PyTorch model, specified as an object registry reference, or a '
                                 ':class:`~alibi_detect.utils.schemas.ModelConfig`.')
    kernel_a: Union[str, KernelConfig] \
        = Field("@utils.tensorflow.kernels.GaussianRBF",
                description='The kernel to apply to the projected inputs. Defaults to a '
                            ':class:`~alibi_detect.utils.tensorflow.kernels.GaussianRBF` with trainable bandwidth.')
    kernel_b: Optional[Union[str, KernelConfig]] \
        = Field("@utils.tensorflow.kernels.GaussianRBF",
                description='The kernel to apply to the raw inputs. Defaults to a '
                            ':class:`~alibi_detect.utils.tensorflow.kernels.GaussianRBF` with trainable bandwidth. '
                            'Set to `None` in order to use only the deep component (i.e. `eps=0`).')
    eps: Union[float, str] \
        = Field('trainable', description="The proportion (in [0,1]) of weight to assign to the kernel applied to raw "
                                         "inputs. This can be either specified or set to `'trainable'`. Only relevant "
                                         "is `kernel_b` is not `None`.")


class DeepKernelConfigResolved(CustomBaseModel):
    """
    Resolved schema for :class:`~alibi_detect.utils.tensorflow.kernels.DeepKernel`'s.
    """
    proj: SupportedModels_py = Field(..., description='The projection to be applied to the inputs before applying '
                                                      '`kernel_a`. This should be a Tensorflow or PyTorch model.')
    kernel_a: Union[Callable, KernelConfigResolved] \
        = Field(..., description='The kernel to apply to the projected inputs.')
    kernel_b: Optional[Union[Callable, KernelConfigResolved]] \
        = Field(..., description='The kernel to apply to the raw inputs. Set to None in order to use only the deep '
                                 'component (i.e. eps=0).')
    # TODO - would be good to set kernel defaults to GaussianRBF(trainable=True). But not clear
    #  how to do this and handle TensorFlow vs PyTorch (especially w/ optional deps)
    eps: Union[float, str]  \
        = Field('trainable', description="The proportion (in [0,1]) of weight to assign to the kernel applied to raw "
                                         "inputs. This can be either specified or set to `'trainable'`. Only relevant "
                                         "is `kernel_b` is not `None`.")


class DriftDetectorConfig(DetectorConfig):
    """
    Unresolved base schema for drift detectors.
    """
    # args/kwargs shared by all drift detectors
    x_ref: str = Field(..., description='Data used as reference distribution. Should be a string referencing a '
                                        'NumPy `.npy` file.')
    p_val: float = Field(.05, description='p-value threshold used for significance of the statistical test.')
    x_ref_preprocessed: bool \
        = Field(False, description='Whether or not the reference data x_ref has already been preprocessed. If True, '
                                   'the reference data will be skipped and preprocessing will only be applied to the '
                                   'test data passed to predict.')
    preprocess_fn: Optional[Union[str, PreprocessConfig]] \
        = Field(None, description='Function to preprocess the data before computing the data drift metrics. Either a '
                                  'string referencing a serialized function in `.dill` format, an object registry '
                                  'reference, or a :class:`~alibi_detect.utils.schemas.PreprocessConfig`.')
    input_shape: Optional[tuple] = Field(None, description='Optionally pass the shape of the input data. Used when '
                                                           'saving detectors.')
    data_type: Optional[str] = Field(None, description="Specify data type added to the metadata. E.g. `‘tabular’` "
                                                       "or `‘image’`.")


class DriftDetectorConfigResolved(DetectorConfig):
    """
    Resolved base schema for drift detectors.
    """
    # args/kwargs shared by all drift detectors
    x_ref: Union[np.ndarray, list] = Field(..., description='Data used as reference distribution.')
    p_val: float = Field(.05, description='p-value threshold used for significance of the statistical test.')
    x_ref_preprocessed: bool \
        = Field(False, description='Whether or not the reference data x_ref has already been preprocessed. If True, '
                                   'the reference data will be skipped and preprocessing will only be applied to the '
                                   'test data passed to predict.')
    preprocess_fn: Optional[Union[Callable, PreprocessConfigResolved]] \
        = Field(None, description='Function to preprocess the data before computing the data drift metrics.')
    input_shape: Optional[tuple] = Field(None, description='Optionally pass the shape of the input data. Used when '
                                                           'saving detectors.')
    data_type: Optional[str] = Field(None, description="Specify data type added to the metadata. E.g. `‘tabular’` "
                                                       "or `‘image’`.")


class KSDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `KSDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/ksdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.KSDrift` documentation for a description of each field.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class KSDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `KSDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/ksdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.KSDrift` documentation for a description of each field.
    Resolved schema for the :class:`~alibi_detect.cd.KSDrift` detector.
    """
    preprocess_at_init: bool = True  # Note: Duplication needed to avoid mypy error (unless we allow reassignment)
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class ChiSquareDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `ChiSquareDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/chisquaredrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ChiSquareDrift` documentation for a description of each field.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Union[int, List[int]]] = None
    n_features: Optional[int] = None


class ChiSquareDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `ChiSquareDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/chisquaredrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ChiSquareDrift` documentation for a description of each field.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Union[int, List[int]]] = None
    n_features: Optional[int] = None


class TabularDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `TabularDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/tabulardrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.TabularDrift` documentation for a description of each field.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Optional[Union[int, List[int]]]] = None
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class TabularDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `TabularDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/tabulardrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.TabularDrift` documentation for a description of each field.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    categories_per_feature: Dict[int, Optional[Union[int, List[int]]]] = None
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class CVMDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `CVMDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/cvmdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.CVMDrift` documentation for a description of each field.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    n_features: Optional[int] = None


class CVMDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `CVMDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/cvmdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.CVMDrift` documentation for a description of each field.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    n_features: Optional[int] = None


class FETDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `FETDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/fetdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.FETDrift` documentation for a description of each field.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class FETDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `FETDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/fetdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.FETDrift` documentation for a description of each field.
    """
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: str = 'bonferroni'
    alternative: str = 'two-sided'
    n_features: Optional[int] = None


class MMDDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `MMDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.MMDDrift` documentation for a description of each field.
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
    Resolved schema for the
    `MMDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.MMDDrift` documentation for a description of each field.
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
    Unresolved schema for the
    `LSDDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/lsdddrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.LSDDDrift` documentation for a description of each field.
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
    Resolved schema for the
    `LSDDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/lsdddrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.LSDDDrift` documentation for a description of each field.
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
    Unresolved schema for the
    `ClassifierDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ClassifierDrift` documentation for a description of each field.
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
    Resolved schema for the
    `ClassifierDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ClassifierDrift` documentation for a description of each field.
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
    Unresolved schema for the
    `SpotTheDiffDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/spotthediffdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.SpotTheDiffDrift` documentation for a description of each field.
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
    Resolved schema for the
    `SpotTheDiffDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/spotthediffdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.SpotTheDiffDrift` documentation for a description of each field.
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
    Unresolved schema for the
    `LearnedKernelDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/learnedkerneldrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.LearnedKernelDrift` documentation for a description of each field.
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
    Resolved schema for the
    `LearnedKernelDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/learnedkerneldrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.LearnedKernelDrift` documentation for a description of each field.
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
}  # type:


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
