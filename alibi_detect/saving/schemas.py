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
from typing import Callable, Dict, List, Optional, Type, Union, Any

import numpy as np
from pydantic import BaseModel, validator

from alibi_detect.utils.frameworks import Framework
from alibi_detect.utils._types import (Literal, supported_models_all, supported_models_tf,
                                       supported_models_sklearn, supported_models_torch, supported_optimizers_tf,
                                       supported_optimizers_torch, supported_optimizers_all)
from alibi_detect.saving.validators import NDArray, validate_framework, coerce_int2list, coerce_2_tensor


class SupportedModel:
    """
    Pydantic custom type to check the model is one of the supported types (conditional on what optional deps
    are installed).
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_model

    @classmethod
    def validate_model(cls, model: Any, values: dict) -> Any:
        backend = values['backend']
        err_msg = f"`backend={backend}` but the `model` doesn't appear to be a {backend} supported model, "\
                  f"or {backend} is not installed. Model: {model}"
        if backend == Framework.TENSORFLOW and not isinstance(model, supported_models_tf):
            raise TypeError(err_msg)
        elif backend == Framework.PYTORCH and not isinstance(model, supported_models_torch):
            raise TypeError(err_msg)
        elif backend == Framework.SKLEARN and not isinstance(model, supported_models_sklearn):
            raise TypeError(f"`backend={backend}` but the `model` doesn't appear to be a {backend} supported model.")
        elif isinstance(model, supported_models_all):  # If model supported and no `backend` incompatibility
            return model
        else:  # Catch any other unexpected issues
            raise TypeError('The model is not recognised as a supported type.')


class SupportedOptimizer:
    """
    Pydantic custom type to check the optimizer is one of the supported types (conditional on what optional deps
    are installed).
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_optimizer

    @classmethod
    def validate_optimizer(cls, optimizer: Any, values: dict) -> Any:
        backend = values['backend']
        err_msg = f"`backend={backend}` but the `optimizer` doesn't appear to be a {backend} supported optimizer, "\
                  f"or {backend} is not installed. Optimizer: {optimizer}"
        if backend == Framework.TENSORFLOW and not isinstance(optimizer, supported_optimizers_tf):
            raise TypeError(err_msg)
        elif backend == Framework.PYTORCH and not isinstance(optimizer, supported_optimizers_torch):
            raise TypeError(err_msg)
        elif isinstance(optimizer, supported_optimizers_all):  # If optimizer supported and no `backend` incompatibility
            return optimizer
        else:  # Catch any other unexpected issues
            raise TypeError('The model is not recognised as a supported type.')


# TODO - We could add validator to check `model` and `embedding` type when chained together. Leave this until refactor
#  of preprocess_drift.


class SupportedDevice:
    """
    Pydantic custom type to check the device is correct for the choice of backend (conditional on what optional deps
    are installed).
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_device

    @classmethod
    def validate_device(cls, device: Any, values: dict) -> Any:
        backend = values['backend']
        if backend == Framework.TENSORFLOW or backend == Framework.SKLEARN:
            if device is not None:
                raise TypeError('`device` should not be specified for TensorFlow or Sklearn backends. Leave as `None`.')
            else:
                return device
        elif backend == Framework.PYTORCH or backend == Framework.KEOPS:
            device_str = str(device).split(':')[0]
            if device_str not in ['cpu', 'cuda', 'gpu']:
                raise TypeError(f'`device` should be one of `cpu`, `cuda`, `gpu` or a torch.Device. Got {device}.')
            else:
                return device
        else:  # Catch any other unexpected issues
            raise TypeError('The device is not recognised as a supported type.')


# Custom BaseModel so that we can set default config
class CustomBaseModel(BaseModel):
    """
    Base pydantic model schema. The default pydantic settings are set here.
    """
    class Config:
        arbitrary_types_allowed = True  # since we have np.ndarray's etc
        extra = 'forbid'  # Forbid extra fields so that we catch misspelled fields


# Custom BaseModel with additional kwarg's allowed
class CustomBaseModelWithKwargs(BaseModel):
    """
    Base pydantic model schema. The default pydantic settings are set here.
    """
    class Config:
        arbitrary_types_allowed = True  # since we have np.ndarray's etc
        extra = 'allow'  # Allow extra fields


class MetaData(CustomBaseModel):
    version: str
    version_warning: bool = False


class DetectorConfig(CustomBaseModel):
    """
    Base detector config schema. Only fields universal across all detectors are defined here.
    """
    name: str
    "Name of the detector e.g. `MMDDrift`."
    meta: Optional[MetaData] = None
    "Config metadata. Should not be edited."
    # Note: Although not all detectors have a backend, we define in base class as `backend` also determines
    #  whether tf or torch models used for preprocess_fn.
    # backend validation (only applied if the detector config has a `backend` field
    _validate_backend = validator('backend', allow_reuse=True, pre=False, check_fields=False)(validate_framework)


class ModelConfig(CustomBaseModel):
    """
    Unresolved schema for (ML) models. Note that the model "backend" e.g. 'tensorflow', 'pytorch', 'sklearn', is set
    by `backend` in :class:`DetectorConfig`.

    Examples
    --------
    A TensorFlow classifier model stored in the `model/` directory, with the softmax layer extracted:

    .. code-block :: toml

        [model]
        flavour = "tensorflow"
        src = "model/"
        layer = -1
    """
    flavour: Literal['tensorflow', 'pytorch', 'sklearn']
    """
    Whether the model is a `tensorflow`, `pytorch` or `sklearn` model. XGBoost models following the scikit-learn API
    are also included under `sklearn`.
    """
    src: str
    """
    Filepath to directory storing the model (relative to the `config.toml` file, or absolute). At present,
    TensorFlow models must be stored in
    `H5 format <https://www.tensorflow.org/guide/keras/save_and_serialize#keras_h5_format>`_.
    """
    custom_objects: Optional[dict] = None
    """
    Dictionary of custom objects. Passed to the tensorflow
    `load_model <https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model>`_ function. This can be
    used to pass custom registered functions and classes to a model.
    """
    layer: Optional[int] = None
    """
    Optional index of hidden layer to extract. If not `None`, a
    :class:`~alibi_detect.cd.tensorflow.preprocess.HiddenOutput` or
    :class:`~alibi_detect.cd.pytorch.preprocess.HiddenOutput` model is returned (dependent on `flavour`).
    Only applies to 'tensorflow' and 'pytorch' models.
    """
    # Validators
    _validate_flavour = validator('flavour', allow_reuse=True, pre=False)(validate_framework)


class EmbeddingConfig(CustomBaseModel):
    """
    Unresolved schema for text embedding models. Currently, only pre-trained
    `HuggingFace transformer <https://github.com/huggingface/transformers>`_ models are supported.

    Examples
    --------
    Using the hidden states at the output of each layer of a TensorFlow
    `BERT base <https://huggingface.co/bert-base-cased>`_ model as text embeddings:

    .. code-block :: toml

        [embedding]
        flavour = "tensorflow"
        src = "bert-base-cased"
        type = "hidden_state"
        layers = [-1, -2, -3, -4, -5, -6, -7, -8]
    """
    flavour: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    """
    Whether the embedding model is a `tensorflow` or `pytorch` model.
    """
    type: Literal['pooler_output', 'last_hidden_state', 'hidden_state', 'hidden_state_cls']
    """
    The type of embedding to be loaded. See `embedding_type` in
    :class:`~alibi_detect.models.tensorflow.embedding.TransformerEmbedding`.
    """
    layers: Optional[List[int]] = None
    "List specifying the hidden layers to be used to extract the embedding."
    # TODO - add check conditional on embedding type (see docstring in above)
    src: str
    """
    Model name e.g. `"bert-base-cased"`, or a filepath to directory storing the model to extract embeddings from
    (relative to the `config.toml` file, or absolute).
    """
    # Validators
    _validate_flavour = validator('flavour', allow_reuse=True, pre=False)(validate_framework)


class TokenizerConfig(CustomBaseModel):
    """
    Unresolved schema for text tokenizers. Currently, only pre-trained
    `HuggingFace tokenizer <https://github.com/huggingface/tokenizers>`_ models are supported.

    Examples
    --------
    `BERT base <https://huggingface.co/bert-base-cased>`_ tokenizer with additional keyword arguments passed to the
    HuggingFace :meth:`~transformers.AutoTokenizer.from_pretrained` method:

     .. code-block :: toml

        [tokenizer]
        src = "bert-base-cased"

        [tokenizer.kwargs]
        use_fast = false
        force_download = true
    """
    src: str
    """
    Model name e.g. `"bert-base-cased"`, or a filepath to directory storing the tokenizer model (relative to the
    `config.toml` file, or absolute). Passed to passed to :meth:`transformers.AutoTokenizer.from_pretrained`.
    """
    kwargs: Optional[dict] = {}
    "Dictionary of keyword arguments to pass to :meth:`transformers.AutoTokenizer.from_pretrained`."


class PreprocessConfig(CustomBaseModel):
    """
    Unresolved schema for drift detector preprocess functions, to be passed to a detector's `preprocess_fn` kwarg.
    Once loaded, the function is wrapped in a :func:`~functools.partial`, to be evaluated within the detector.

    If `src` specifies a generic Python function, the dictionary specified by `kwargs` is passed to it. Otherwise,
    if `src` specifies :func:`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift`
    (`src='@cd.tensorflow.preprocess.preprocess_drift'`), all fields (except `kwargs`) are passed to it.

    Examples
    --------
    Preprocessor with a `model`, text `embedding` and `tokenizer` passed to
    :func:`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift`:

    .. code-block :: toml

        [preprocess_fn]
        src = "@cd.tensorflow.preprocess.preprocess_drift"
        batch_size = 32
        max_len = 100
        tokenizer.src = "tokenizer/"  # TokenizerConfig

        [preprocess_fn.model]
        # ModelConfig
        src = "model/"

        [preprocess_fn.embedding]
        # EmbeddingConfig
        src = "embedding/"
        type = "hidden_state"
        layers = [-1, -2, -3, -4, -5, -6, -7, -8]

    A serialized Python function with keyword arguments passed to it:

    .. code-block :: toml

        [preprocess_fn]
        src = 'myfunction.dill'
        kwargs = {'kwarg1'=0.7, 'kwarg2'=true}
    """
    src: str = "@cd.tensorflow.preprocess.preprocess_drift"
    """
    The preprocessing function. A string referencing a filepath to a serialized function in `dill` format, or an
    object registry reference.
    """

    # Below kwargs are only passed if src == @preprocess_drift
    model: Optional[Union[str, ModelConfig]] = None
    """
    Model used for preprocessing. Either an object registry reference, or a
    :class:`~alibi_detect.utils.schemas.ModelConfig`.
    """

    # TODO - make model required field when src is preprocess_drift
    embedding: Optional[Union[str, EmbeddingConfig]] = None
    """
    A text embedding model. Either a string referencing a HuggingFace transformer model name, an object registry
    reference, or a :class:`~alibi_detect.utils.schemas.EmbeddingConfig`. If `model=None`, the `embedding` is passed to
    :func:`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift` as `model`. Otherwise, the `model` is chained to
    the output of the `embedding` as an additional preprocessing step.
    """
    tokenizer: Optional[Union[str, TokenizerConfig]] = None
    """
    Optional tokenizer for text drift. Either a string referencing a HuggingFace tokenizer model name, or a
    :class:`~alibi_detect.utils.schemas.TokenizerConfig`.
    """
    device: Optional[Literal['cpu', 'cuda', 'gpu']] = None
    """
    Device type used. The default `None` tries to use the GPU and falls back on CPU if needed. Only relevant if
    `src='@cd.torch.preprocess.preprocess_drift'`
    """
    preprocess_batch_fn: Optional[str] = None
    """
    Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed
    by the `model`.
    """
    max_len: Optional[int] = None
    "Optional max token length for text drift."
    batch_size: Optional[int] = int(1e10)
    "Batch size used during prediction."
    dtype: str = 'np.float32'
    "Model output type, e.g. `'tf.float32'`"

    # Additional kwargs
    kwargs: dict = {}
    """
    Dictionary of keyword arguments to be passed to the function specified by `src`. Only used if `src` specifies a
    generic Python function.
    """


class KernelConfig(CustomBaseModelWithKwargs):
    """
    Unresolved schema for kernels, to be passed to a detector's `kernel` kwarg.

    If `src` specifies a :class:`~alibi_detect.utils.tensorflow.GaussianRBF` kernel, the `sigma`, `trainable` and
    `init_sigma_fn` fields are passed to it. Otherwise, all fields except `src` are passed as kwargs.

    Examples
    --------
    A :class:`~alibi_detect.utils.tensorflow.GaussianRBF` kernel, with three different bandwidths:

    .. code-block :: toml

        [kernel]
        src = "@alibi_detect.utils.tensorflow.GaussianRBF"
        trainable = false
        sigma = [0.1, 0.2, 0.3]

    A serialized kernel with keyword arguments passed:

    .. code-block :: toml

        [kernel]
        src = "mykernel.dill"
        sigma = 0.42
        custom_setting = "xyz"
    """
    src: str
    "A string referencing a filepath to a serialized kernel in `.dill` format, or an object registry reference."

    # Below kwargs are only passed if kernel == @GaussianRBF
    flavour: Literal['tensorflow', 'pytorch', 'keops']
    """
    Whether the kernel is a `tensorflow` or `pytorch` kernel.
    """
    sigma: Optional[Union[float, List[float]]] = None
    """
    Bandwidth used for the kernel. Needn’t be specified if being inferred or trained. Can pass multiple values to eval
    kernel with and then average.
    """
    trainable: bool = False
    "Whether or not to track gradients w.r.t. sigma to allow it to be trained."

    init_sigma_fn: Optional[str] = None
    """
    Function used to compute the bandwidth `sigma`. Used when `sigma` is to be inferred. The function's signature
    should match :py:func:`~alibi_detect.utils.tensorflow.kernels.sigma_median`. If `None`, it is set to
    :func:`~alibi_detect.utils.tensorflow.kernels.sigma_median`.
    """
    # Validators
    _validate_flavour = validator('flavour', allow_reuse=True, pre=False)(validate_framework)
    _coerce_sigma2tensor = validator('sigma', allow_reuse=True, pre=False)(coerce_2_tensor)


class DeepKernelConfig(CustomBaseModel):
    """
    Unresolved schema for :class:`~alibi_detect.utils.tensorflow.kernels.DeepKernel`'s.

    Examples
    --------
    A :class:`~alibi_detect.utils.tensorflow.DeepKernel`, with a trainable
    :class:`~alibi_detect.utils.tensorflow.GaussianRBF` kernel applied to the projected inputs and a custom
    serialized kernel applied to the raw inputs:

    .. code-block :: toml

        [kernel]
        eps = 0.01

        [kernel.kernel_a]
        src = "@utils.tensorflow.kernels.GaussianRBF"
        trainable = true

        [kernel.kernel_b]
        src = "custom_kernel.dill"
        sigma = [ 1.2,]
        trainable = false

        [kernel.proj]
        src = "model/"
    """
    proj: Union[str, ModelConfig]
    """
    The projection to be applied to the inputs before applying `kernel_a`. This should be a Tensorflow or PyTorch
    model, specified as an object registry reference, or a :class:`~alibi_detect.utils.schemas.ModelConfig`.
    """
    kernel_a: Union[str, KernelConfig] = "@utils.tensorflow.kernels.GaussianRBF"
    """
    The kernel to apply to the projected inputs. Defaults to a
    :class:`~alibi_detect.utils.tensorflow.kernels.GaussianRBF` with trainable bandwidth.
    """
    kernel_b: Optional[Union[str, KernelConfig]] = "@utils.tensorflow.kernels.GaussianRBF"
    """
    The kernel to apply to the raw inputs. Defaults to a :class:`~alibi_detect.utils.tensorflow.kernels.GaussianRBF`
    with trainable bandwidth. Set to `None` in order to use only the deep component (i.e. `eps=0`).
    """
    eps: Union[float, str] = 'trainable'
    """
    The proportion (in [0,1]) of weight to assign to the kernel applied to raw inputs. This can be either specified or
    set to `'trainable'`. Only relevant is `kernel_b` is not `None`.
    """


class OptimizerConfig(CustomBaseModelWithKwargs):
    """
    Unresolved schema for optimizers. The `optimizer` dictionary has two possible formats:

    1. A configuration dictionary compatible with
    `tf.keras.optimizers.deserialize <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/deserialize>`_.
    For `backend='tensorflow'` only.
    2. A dictionary containing only `class_name`, where this is a string referencing the optimizer name e.g.
    `optimizer.class_name = 'Adam'`. In this case, the tensorflow or pytorch optimizer class of the same name is
    loaded. For `backend='tensorflow'` and `backend='pytorch'`.

    Examples
    --------
    A TensorFlow Adam optimizer:

    .. code-block :: toml

        [optimizer]
        class_name = "Adam"

        [optimizer.config]
        name = "Adam"
        learning_rate = 0.001
        decay = 0.0

    A PyTorch Adam optimizer:

    .. code-block :: toml

        [optimizer]
        class_name = "Adam"
    """
    class_name: str
    config: Optional[Dict[str, Any]] = None


class DriftDetectorConfig(DetectorConfig):
    """
    Unresolved base schema for drift detectors.
    """
    # args/kwargs shared by all drift detectors
    x_ref: str
    "Data used as reference distribution. Should be a string referencing a NumPy `.npy` file."
    preprocess_fn: Optional[Union[str, PreprocessConfig]] = None
    """
    Function to preprocess the data before computing the data drift metrics. A string referencing a serialized function
    in `.dill` format, an object registry reference, or a :class:`~alibi_detect.utils.schemas.PreprocessConfig`.
    """
    input_shape: Optional[tuple] = None
    "Optionally pass the shape of the input data. Used when saving detectors."
    data_type: Optional[str] = None
    "Specify data type added to the metadata. E.g. `‘tabular’`or `‘image’`."
    x_ref_preprocessed: bool = False
    """
    Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test
    data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be
    preprocessed.
    """


class DriftDetectorConfigResolved(DetectorConfig):
    """
    Resolved base schema for drift detectors.
    """
    # args/kwargs shared by all drift detectors
    x_ref: Union[np.ndarray, list]
    "Data used as reference distribution."
    preprocess_fn: Optional[Callable] = None
    "Function to preprocess the data before computing the data drift metrics."
    input_shape: Optional[tuple] = None
    "Optionally pass the shape of the input data. Used when saving detectors."
    data_type: Optional[str] = None
    "Specify data type added to the metadata. E.g. `‘tabular’` or `‘image’`."
    x_ref_preprocessed: bool = False
    """
    Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test
    data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be
    preprocessed.
    """


class KSDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `KSDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/ksdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.KSDrift` documentation for a description of each field.
    """
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: Literal['bonferroni', 'fdr'] = 'bonferroni'
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
    n_features: Optional[int] = None


class KSDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `KSDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/ksdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.KSDrift` documentation for a description of each field.
    Resolved schema for the :class:`~alibi_detect.cd.KSDrift` detector.
    """
    p_val: float = .05
    preprocess_at_init: bool = True  # Note: Duplication needed to avoid mypy error (unless we allow reassignment)
    update_x_ref: Optional[Dict[str, int]] = None
    correction: Literal['bonferroni', 'fdr'] = 'bonferroni'
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
    n_features: Optional[int] = None


class ChiSquareDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `ChiSquareDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/chisquaredrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ChiSquareDrift` documentation for a description of each field.
    """
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: Literal['bonferroni', 'fdr'] = 'bonferroni'
    categories_per_feature: Dict[int, Union[int, List[int]]] = None
    n_features: Optional[int] = None


class ChiSquareDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `ChiSquareDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/chisquaredrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ChiSquareDrift` documentation for a description of each field.
    """
    p_val: float = .05
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
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: Literal['bonferroni', 'fdr'] = 'bonferroni'
    categories_per_feature: Dict[int, Optional[Union[int, List[int]]]] = None
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
    n_features: Optional[int] = None


class TabularDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `TabularDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/tabulardrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.TabularDrift` documentation for a description of each field.
    """
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: Literal['bonferroni', 'fdr'] = 'bonferroni'
    categories_per_feature: Dict[int, Optional[Union[int, List[int]]]] = None
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
    n_features: Optional[int] = None


class CVMDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `CVMDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/cvmdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.CVMDrift` documentation for a description of each field.
    """
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: Literal['bonferroni', 'fdr'] = 'bonferroni'
    n_features: Optional[int] = None


class CVMDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `CVMDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/cvmdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.CVMDrift` documentation for a description of each field.
    """
    p_val: float = .05
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
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: Literal['bonferroni', 'fdr'] = 'bonferroni'
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
    n_features: Optional[int] = None


class FETDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `FETDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/fetdrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.FETDrift` documentation for a description of each field.
    """
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    correction: Literal['bonferroni', 'fdr'] = 'bonferroni'
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
    n_features: Optional[int] = None


class MMDDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `MMDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.MMDDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch', 'keops'] = 'tensorflow'
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    kernel: Optional[Union[str, KernelConfig]] = None
    sigma: Optional[NDArray[np.float32]] = None
    configure_kernel_from_x_ref: bool = True
    n_permutations: int = 100
    batch_size_permutations: int = 1000000
    device: Optional[SupportedDevice] = None


class MMDDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `MMDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.MMDDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch', 'keops'] = 'tensorflow'
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    kernel: Optional[Callable] = None
    sigma: Optional[NDArray[np.float32]] = None
    configure_kernel_from_x_ref: bool = True
    n_permutations: int = 100
    batch_size_permutations: int = 1000000
    device: Optional[SupportedDevice] = None


class LSDDDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `LSDDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/lsdddrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.LSDDDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    sigma: Optional[NDArray[np.float32]] = None
    n_permutations: int = 100
    n_kernel_centers: Optional[int] = None
    lambda_rd_max: float = 0.2
    device: Optional[SupportedDevice] = None


class LSDDDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `LSDDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/lsdddrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.LSDDDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    sigma: Optional[NDArray[np.float32]] = None
    n_permutations: int = 100
    n_kernel_centers: Optional[int] = None
    lambda_rd_max: float = 0.2
    device: Optional[SupportedDevice] = None


class ClassifierDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `ClassifierDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ClassifierDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch', 'sklearn'] = 'tensorflow'
    p_val: float = .05
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
    optimizer: Optional[Union[str, OptimizerConfig]] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[str] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: Optional[str] = None
    device: Optional[SupportedDevice] = None
    dataloader: Optional[str] = None  # TODO: placeholder, will need to be updated for pytorch implementation
    use_calibration: bool = False
    calibration_kwargs: Optional[dict] = None
    use_oob: bool = False


class ClassifierDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `ClassifierDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ClassifierDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch', 'sklearn'] = 'tensorflow'
    p_val: float = .05
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    model: Optional[SupportedModel] = None
    preds_type: Literal['probs', 'logits'] = 'probs'
    binarize_preds: bool = False
    reg_loss_fn: Optional[Callable] = None
    train_size: Optional[float] = .75
    n_folds: Optional[int] = None
    retrain_from_scratch: bool = True
    seed: int = 0
    optimizer: Optional[SupportedOptimizer] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[Callable] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: Optional[Callable] = None
    device: Optional[SupportedDevice] = None
    dataloader: Optional[Callable] = None  # TODO: placeholder, will need to be updated for pytorch implementation
    use_calibration: bool = False
    calibration_kwargs: Optional[dict] = None
    use_oob: bool = False


class SpotTheDiffDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `SpotTheDiffDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/spotthediffdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.SpotTheDiffDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    p_val: float = .05
    binarize_preds: bool = False
    train_size: Optional[float] = .75
    n_folds: Optional[int] = None
    retrain_from_scratch: bool = True
    seed: int = 0
    optimizer: Optional[Union[str, OptimizerConfig]] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[str] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: Optional[str] = None
    kernel: Optional[Union[str, KernelConfig]] = None
    n_diffs: int = 1
    initial_diffs: Optional[str] = None
    l1_reg: float = 0.01
    device: Optional[SupportedDevice] = None
    dataloader: Optional[str] = None  # TODO: placeholder, will need to be updated for pytorch implementation


class SpotTheDiffDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `SpotTheDiffDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/spotthediffdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.SpotTheDiffDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    p_val: float = .05
    binarize_preds: bool = False
    train_size: Optional[float] = .75
    n_folds: Optional[int] = None
    retrain_from_scratch: bool = True
    seed: int = 0
    optimizer: Optional[SupportedOptimizer] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    preprocess_batch_fn: Optional[Callable] = None
    epochs: int = 3
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: Optional[Callable] = None
    kernel: Optional[Callable] = None
    n_diffs: int = 1
    initial_diffs: Optional[np.ndarray] = None
    l1_reg: float = 0.01
    device: Optional[SupportedDevice] = None
    dataloader: Optional[Callable] = None  # TODO: placeholder, will need to be updated for pytorch implementation


class LearnedKernelDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `LearnedKernelDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/learnedkerneldrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.LearnedKernelDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch', 'keops'] = 'tensorflow'
    p_val: float = .05
    kernel: Union[str, DeepKernelConfig]
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    n_permutations: int = 100
    batch_size_permutations: int = 1000000
    var_reg: float = 1e-5
    reg_loss_fn: Optional[str] = None
    train_size: Optional[float] = .75
    retrain_from_scratch: bool = True
    optimizer: Optional[Union[str, OptimizerConfig]] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    batch_size_predict: int = 1000000
    preprocess_batch_fn: Optional[str] = None
    epochs: int = 3
    num_workers: int = 0
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: Optional[str] = None
    device: Optional[SupportedDevice] = None
    dataloader: Optional[str] = None  # TODO: placeholder, will need to be updated for pytorch implementation


class LearnedKernelDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `LearnedKernelDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/learnedkerneldrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.LearnedKernelDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch', 'keops'] = 'tensorflow'
    p_val: float = .05
    kernel: Optional[Callable] = None
    preprocess_at_init: bool = True
    update_x_ref: Optional[Dict[str, int]] = None
    n_permutations: int = 100
    batch_size_permutations: int = 1000000
    var_reg: float = 1e-5
    reg_loss_fn: Optional[Callable] = None
    train_size: Optional[float] = .75
    retrain_from_scratch: bool = True
    optimizer: Optional[SupportedOptimizer] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    batch_size_predict: int = 1000000
    preprocess_batch_fn: Optional[Callable] = None
    epochs: int = 3
    num_workers: int = 0
    verbose: int = 0
    train_kwargs: Optional[dict] = None
    dataset: Optional[Callable] = None
    device: Optional[SupportedDevice] = None
    dataloader: Optional[Callable] = None  # TODO: placeholder, will need to be updated for pytorch implementation


class ContextMMDDriftConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `ContextMMDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/contextmmddrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ContextMMDDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    p_val: float = .05
    c_ref: str
    preprocess_at_init: bool = True
    update_ref: Optional[Dict[str, int]] = None
    x_kernel: Optional[Union[str, KernelConfig]] = None
    c_kernel: Optional[Union[str, KernelConfig]] = None
    n_permutations: int = 100
    prop_c_held: float = 0.25
    n_folds: int = 5
    batch_size: Optional[int] = 256
    verbose: bool = False
    device: Optional[SupportedDevice] = None


class ContextMMDDriftConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `MMDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html>`_ detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.MMDDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    p_val: float = .05
    c_ref: np.ndarray
    preprocess_at_init: bool = True
    update_ref: Optional[Dict[str, int]] = None
    x_kernel: Optional[Callable] = None
    c_kernel: Optional[Callable] = None
    n_permutations: int = 100
    prop_c_held: float = 0.25
    n_folds: int = 5
    batch_size: Optional[int] = 256
    verbose: bool = False
    device: Optional[SupportedDevice] = None


class MMDDriftOnlineConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `MMDDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinemmddrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.MMDDriftOnline` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    ert: float
    window_size: int
    kernel: Optional[Union[str, KernelConfig]] = None
    sigma: Optional[np.ndarray] = None
    n_bootstraps: int = 1000
    device: Optional[SupportedDevice] = None
    verbose: bool = True


class MMDDriftOnlineConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `MMDDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinemmddrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.MMDDriftOnline` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    ert: float
    window_size: int
    kernel: Optional[Callable] = None
    sigma: Optional[np.ndarray] = None
    n_bootstraps: int = 1000
    device: Optional[SupportedDevice] = None
    verbose: bool = True


class LSDDDriftOnlineConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `LSDDDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinelsdddrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.LSDDDriftOnline` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    ert: float
    window_size: int
    sigma: Optional[np.ndarray] = None
    n_bootstraps: int = 1000
    n_kernel_centers: Optional[int] = None
    lambda_rd_max: float = 0.2
    device: Optional[SupportedDevice] = None
    verbose: bool = True


class LSDDDriftOnlineConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `LSDDDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinelsdddrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.LSDDDriftOnline` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    ert: float
    window_size: int
    sigma: Optional[np.ndarray] = None
    n_bootstraps: int = 1000
    n_kernel_centers: Optional[int] = None
    lambda_rd_max: float = 0.2
    device: Optional[SupportedDevice] = None
    verbose: bool = True


class CVMDriftOnlineConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `CVMDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinecvmdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.CVMDriftOnline` documentation for a description of each field.
    """
    ert: float
    window_sizes: List[int]
    n_bootstraps: int = 10000
    batch_size: int = 64
    n_features: Optional[int] = None
    verbose: bool = True

    # validators
    _coerce_int2list = validator('window_sizes', allow_reuse=True, pre=True)(coerce_int2list)


class CVMDriftOnlineConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `CVMDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinecvmdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.CVMDriftOnline` documentation for a description of each field.
    """
    ert: float
    window_sizes: List[int]
    n_bootstraps: int = 10000
    batch_size: int = 64
    n_features: Optional[int] = None
    verbose: bool = True

    # validators
    _coerce_int2list = validator('window_sizes', allow_reuse=True, pre=True)(coerce_int2list)


class FETDriftOnlineConfig(DriftDetectorConfig):
    """
    Unresolved schema for the
    `FETDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinefetdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.FETDriftOnline` documentation for a description of each field.
    """
    ert: float
    window_sizes: List[int]
    n_bootstraps: int = 10000
    t_max: Optional[int] = None
    alternative: Literal['greater', 'less'] = 'greater'
    lam: float = 0.99
    n_features: Optional[int] = None
    verbose: bool = True

    # validators
    _coerce_int2list = validator('window_sizes', allow_reuse=True, pre=True)(coerce_int2list)


class FETDriftOnlineConfigResolved(DriftDetectorConfigResolved):
    """
    Resolved schema for the
    `FETDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinefetdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.FETDriftOnline` documentation for a description of each field.
    """
    ert: float
    window_sizes: List[int]
    n_bootstraps: int = 10000
    t_max: Optional[int] = None
    alternative: Literal['greater', 'less'] = 'greater'
    lam: float = 0.99
    n_features: Optional[int] = None
    verbose: bool = True

    # validators
    _coerce_int2list = validator('window_sizes', allow_reuse=True, pre=True)(coerce_int2list)


# The uncertainty detectors don't inherit from DriftDetectorConfig since their kwargs are a little different from the
# other drift detectors (e.g. no preprocess_fn). Subject to change in the future.
class ClassifierUncertaintyDriftConfig(DetectorConfig):
    """
    Unresolved schema for the
    `ClassifierUncertaintyDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/modeluncdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ClassifierUncertaintyDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    x_ref: str
    model: Union[str, ModelConfig]
    p_val: float = .05
    x_ref_preprocessed: bool = False
    update_x_ref: Optional[Dict[str, int]] = None
    preds_type: Literal['probs', 'logits'] = 'probs'
    uncertainty_type: Literal['entropy', 'margin'] = 'entropy'
    margin_width: float = 0.1
    batch_size: int = 32
    preprocess_batch_fn: Optional[str] = None
    device: Optional[SupportedDevice] = None
    tokenizer: Optional[Union[str, TokenizerConfig]] = None
    max_len: Optional[int] = None
    input_shape: Optional[tuple] = None
    data_type: Optional[str] = None


class ClassifierUncertaintyDriftConfigResolved(DetectorConfig):
    """
    Resolved schema for the
    `ClassifierUncertaintyDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/modeluncdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.ClassifierUncertaintyDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    x_ref: Union[np.ndarray, list]
    model: Optional[SupportedModel] = None
    p_val: float = .05
    x_ref_preprocessed: bool = False
    update_x_ref: Optional[Dict[str, int]] = None
    preds_type: Literal['probs', 'logits'] = 'probs'
    uncertainty_type: Literal['entropy', 'margin'] = 'entropy'
    margin_width: float = 0.1
    batch_size: int = 32
    preprocess_batch_fn: Optional[Callable] = None
    device: Optional[SupportedDevice] = None
    tokenizer: Optional[Union[str, Callable]] = None
    max_len: Optional[int] = None
    input_shape: Optional[tuple] = None
    data_type: Optional[str] = None


class RegressorUncertaintyDriftConfig(DetectorConfig):
    """
    Unresolved schema for the
    `RegressorUncertaintyDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/modeluncdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.RegressorUncertaintyDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    x_ref: str
    model: Union[str, ModelConfig]
    p_val: float = .05
    x_ref_preprocessed: bool = False
    update_x_ref: Optional[Dict[str, int]] = None
    uncertainty_type: Literal['mc_dropout', 'ensemble'] = 'mc_dropout'
    n_evals: int = 25
    batch_size: int = 32
    preprocess_batch_fn: Optional[str] = None
    device: Optional[SupportedDevice] = None
    tokenizer: Optional[Union[str, TokenizerConfig]] = None
    max_len: Optional[int] = None
    input_shape: Optional[tuple] = None
    data_type: Optional[str] = None


class RegressorUncertaintyDriftConfigResolved(DetectorConfig):
    """
    Resolved schema for the
    `RegressorUncertaintyDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/modeluncdrift.html>`_
    detector.

    Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
    :class:`~alibi_detect.cd.RegressorUncertaintyDrift` documentation for a description of each field.
    """
    backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'
    x_ref: Union[np.ndarray, list]
    model: Optional[SupportedModel] = None
    p_val: float = .05
    x_ref_preprocessed: bool = False
    update_x_ref: Optional[Dict[str, int]] = None
    uncertainty_type: Literal['mc_dropout', 'ensemble'] = 'mc_dropout'
    n_evals: int = 25
    batch_size: int = 32
    preprocess_batch_fn: Optional[Callable] = None
    device: Optional[SupportedDevice] = None
    tokenizer: Optional[Callable] = None
    max_len: Optional[int] = None
    input_shape: Optional[tuple] = None
    data_type: Optional[str] = None


# Unresolved schema dictionary (used in alibi_detect.utils.loading)
DETECTOR_CONFIGS: Dict[str, Type[DetectorConfig]] = {
    'KSDrift': KSDriftConfig,
    'ChiSquareDrift': ChiSquareDriftConfig,
    'TabularDrift': TabularDriftConfig,
    'CVMDrift': CVMDriftConfig,
    'FETDrift': FETDriftConfig,
    'MMDDrift': MMDDriftConfig,
    'LSDDDrift': LSDDDriftConfig,
    'ClassifierDrift': ClassifierDriftConfig,
    'SpotTheDiffDrift': SpotTheDiffDriftConfig,
    'LearnedKernelDrift': LearnedKernelDriftConfig,
    'ContextMMDDrift': ContextMMDDriftConfig,
    'MMDDriftOnline': MMDDriftOnlineConfig,
    'LSDDDriftOnline': LSDDDriftOnlineConfig,
    'CVMDriftOnline': CVMDriftOnlineConfig,
    'FETDriftOnline': FETDriftOnlineConfig,
    'ClassifierUncertaintyDrift': ClassifierUncertaintyDriftConfig,
    'RegressorUncertaintyDrift': RegressorUncertaintyDriftConfig,
}


# Resolved schema dictionary (used in alibi_detect.utils.loading)
DETECTOR_CONFIGS_RESOLVED: Dict[str, Type[DetectorConfig]] = {
    'KSDrift': KSDriftConfigResolved,
    'ChiSquareDrift': ChiSquareDriftConfigResolved,
    'TabularDrift': TabularDriftConfigResolved,
    'CVMDrift': CVMDriftConfigResolved,
    'FETDrift': FETDriftConfigResolved,
    'MMDDrift': MMDDriftConfigResolved,
    'LSDDDrift': LSDDDriftConfigResolved,
    'ClassifierDrift': ClassifierDriftConfigResolved,
    'SpotTheDiffDrift': SpotTheDiffDriftConfigResolved,
    'LearnedKernelDrift': LearnedKernelDriftConfigResolved,
    'ContextMMDDrift': ContextMMDDriftConfigResolved,
    'MMDDriftOnline': MMDDriftOnlineConfigResolved,
    'LSDDDriftOnline': LSDDDriftOnlineConfigResolved,
    'CVMDriftOnline': CVMDriftOnlineConfigResolved,
    'FETDriftOnline': FETDriftOnlineConfigResolved,
    'ClassifierUncertaintyDrift': ClassifierUncertaintyDriftConfigResolved,
    'RegressorUncertaintyDrift': RegressorUncertaintyDriftConfigResolved,
}
