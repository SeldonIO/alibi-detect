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

from typing import Any, Callable, Dict, List, Optional, Type, Union

# TODO - conditional checks depending on backend etc
# TODO - consider validating output of get_config calls
import numpy as np
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from alibi_detect.cd.tensorflow import UAE as UAE_tf
from alibi_detect.cd.tensorflow import HiddenOutput as HiddenOutput_tf
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.utils._types import Literal, NDArray
from alibi_detect.utils.frameworks import has_tensorflow
from alibi_detect.version import __config_spec__, __version__

SupportedModels_list = []
if has_tensorflow:
    import tensorflow as tf
    SupportedModels_list += [tf.keras.Model, UAE_tf, HiddenOutput_tf]
# if has_pytorch:
#    import torch
#    SupportedModels_list.append()  # TODO
# if has_sklearn:
#    import sklearn
#    SupportedModels_list.append()  # TODO
# SupportedModels is a tuple of possible models (conditional on installed deps). This is used in isinstance() etc.
SupportedModels = tuple(SupportedModels_list)
# SupportedModels_types is a typing Union for use with pydantic. We include all optional deps in here so that they
# are all documented in the api docs (where the optional deps are not installed at build time)
SupportedModels_types = Union['tf.keras.Model', UAE_tf, HiddenOutput_tf]


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
    name: str
    "Name of the detector e.g. `MMDDrift`."
    backend: Literal['tensorflow', 'pytorch', 'sklearn'] = 'tensorflow'
    "The detector backend."
    meta: Optional[MetaData] = MetaData()
    "Config metadata. Should not be edited."
    # Note: Although not all detectors have a backend, we define in base class as `backend` also determines
    #  whether tf or torch models used for preprocess_fn.


class ModelConfig(CustomBaseModel):
    """
    Unresolved schema for (ML) models. Note that the model "backend" e.g. 'tensorflow', 'pytorch', 'sklearn', is set
    by `backend` in :class:`DetectorConfig`.

    Examples
    --------
    A TensorFlow classifier model stored in the `model/` directory, with the softmax layer extracted:

    .. code-block :: toml

        [model]
        src = "model/"
        layer = -1
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
    :class:`~alibi_detect.cd.tensorflow.preprocess.HiddenOutput` model is returned.
    """


class EmbeddingConfig(CustomBaseModel):
    """
    Unresolved schema for text embedding models. Currently, only pre-trained
    `HuggingFace transformer <https://github.com/huggingface/transformers>`_ models are supported.

    Examples
    --------
    Using the hidden states at the output of each layer of the
    `BERT base <https://huggingface.co/bert-base-cased>`_ model as text embeddings:

    .. code-block :: toml

        [embedding]
        src = "bert-base-cased"
        type = "hidden_state"
        layers = [-1, -2, -3, -4, -5, -6, -7, -8]
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
    device: Optional[Literal['cpu', 'cuda']] = None
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
    dtype: Optional[str] = None
    "Model output type, e.g. `'tf.float32'`"

    # Additional kwargs
    kwargs: dict = {}
    """
    Dictionary of keyword arguments to be passed to the function specified by `src`. Only used if `src` specifies a
    generic Python function.
    """


class PreprocessConfigResolved(CustomBaseModel):
    """
    Resolved schema for drift detector preprocess functions, to be passed to a detector's `preprocess_fn` kwarg.
    Once loaded, the function is wrapped in a :func:`~functools.partial`, to be evaluated within the detector.

    If `src` is a generic Python function, the dictionary specified by `kwargs` is passed to it. Otherwise,
    if `src` is a :func:`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift` function, all fields
    (except `kwargs`) are passed to it.
    """
    src: Callable
    "The preprocessing function."

    # Below kwargs are only passed if src == @preprocess_drift
    model: Optional[SupportedModels_types] = None
    "Model used for preprocessing."
    embedding: Optional[TransformerEmbedding] = None
    """
    A text embedding model. If `model=None`, the `embedding` is passed to
    :func:`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift` as `model`. Otherwise, the `model` is chained to
    the output of the `embedding` as an additional preprocessing step.')
    """
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    "Optional tokenizer for text drift."
    device: Optional[Any] = None
    """
    Device type used. The default `None` tries to use the GPU and falls back on CPU if needed. Only relevant if
    function is :func:`~alibi_detect.cd.pytorch.preprocess.preprocess_drift`.
    """
    # TODO: Set as Any and None for now. Clarify when pytorch implemented
    preprocess_batch_fn: Optional[Callable] = None
    """
    Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed
    by the `model`.
    """
    max_len: Optional[int] = None
    "Optional max token length for text drift."
    batch_size: Optional[int] = int(1e10)
    "Batch size used during prediction."
    dtype: Optional[Union['tf.DType', np.dtype, type]] = None  # TODO - add pytorch
    "Model output type, e.g. `tf.float32`"

    # Additional kwargs
    kwargs: dict = {}
    """
    Dictionary of keyword arguments to be passed to the function specified by `src`. Only used if `src` specifies a
    generic Python function.
    """


class KernelConfig(CustomBaseModel):
    """
    Unresolved schema for kernels, to be passed to a detector's `kernel` kwarg.

    If `src` specifies a :class:`~alibi_detect.utils.tensorflow.GaussianRBF` kernel, the `sigma` and `trainable` fields
    are passed to it. Otherwise, the `kwargs` field is passed.

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

        [kernel.kwargs]
        sigma = 0.42
        custom_setting = "xyz"
    """
    src: str
    "A string referencing a filepath to a serialized kernel in `.dill` format, or an object registry reference."

    # Below kwargs are only passed if kernel == @GaussianRBF
    sigma: Optional[NDArray[np.float32]] = None
    """
    Bandwidth used for the kernel. Needn’t be specified if being inferred or trained. Can pass multiple values to eval
    kernel with and then average.
    """
    trainable: bool = False
    "Whether or not to track gradients w.r.t. sigma to allow it to be trained."

    # Additional kwargs
    kwargs: dict = {}
    "Dictionary of keyword arguments to pass to the kernel."


class KernelConfigResolved(CustomBaseModel):
    """
    Resolved schema for kernels, to be passed to a detector's `kernel` kwarg.

    If `src` is a :class:`~alibi_detect.utils.tensorflow.GaussianRBF` kernel, the `sigma` and `trainable` fields
    are passed to it. Otherwise, the `kwargs` field is passed.
    """
    src: Callable
    "The kernel."

    # Below kwargs are only passed if kernel == @GaussianRBF
    sigma: Optional[NDArray[np.float32]] = None
    """
    Bandwidth used for the kernel. Needn’t be specified if being inferred or trained. Can pass multiple values to eval
    kernel with and then average.
    """

    trainable: bool = False
    "Whether or not to track gradients w.r.t. sigma to allow it to be trained."

    # Additional kwargs
    kwargs: dict = {}
    "Dictionary of keyword arguments to pass to the kernel."


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


class DeepKernelConfigResolved(CustomBaseModel):
    """
    Resolved schema for :class:`~alibi_detect.utils.tensorflow.kernels.DeepKernel`'s.
    """
    proj: SupportedModels_types
    """
    The projection to be applied to the inputs before applying `kernel_a`. This should be a Tensorflow or PyTorch model.
    """
    kernel_a: Union[Callable, KernelConfigResolved]
    "The kernel to apply to the projected inputs."
    kernel_b: Optional[Union[Callable, KernelConfigResolved]]
    "The kernel to apply to the raw inputs. Set to None in order to use only the deep component (i.e. eps=0)."
    # TODO - would be good to set kernel defaults to GaussianRBF(trainable=True). But not clear
    #  how to do this and handle TensorFlow vs PyTorch (especially w/ optional deps)
    eps: Union[float, str] = 'trainable'
    """
    The proportion (in [0,1]) of weight to assign to the kernel applied to raw inputs. This can be either specified or
    set to `'trainable'`. Only relevant is `kernel_b` is not `None`.
    """


class DriftDetectorConfig(DetectorConfig):
    """
    Unresolved base schema for drift detectors.
    """
    # args/kwargs shared by all drift detectors
    x_ref: str
    "Data used as reference distribution. Should be a string referencing a NumPy `.npy` file."
    p_val: float = .05
    "p-value threshold used for significance of the statistical test."
    x_ref_preprocessed: bool = False
    """
    Whether or not the reference data x_ref has already been preprocessed. If True, the reference data will be skipped
    and preprocessing will only be applied to the test data passed to predict.
    """
    preprocess_fn: Optional[Union[str, PreprocessConfig]] = None
    """
    Function to preprocess the data before computing the data drift metrics. A string referencing a serialized function
    in `.dill` format, an object registry reference, or a :class:`~alibi_detect.utils.schemas.PreprocessConfig`.
    """
    input_shape: Optional[tuple] = None
    "Optionally pass the shape of the input data. Used when saving detectors."
    data_type: Optional[str] = None
    "Specify data type added to the metadata. E.g. `‘tabular’`or `‘image’`."


class DriftDetectorConfigResolved(DetectorConfig):
    """
    Resolved base schema for drift detectors.
    """
    # args/kwargs shared by all drift detectors
    x_ref: Union[np.ndarray, list]
    "Data used as reference distribution."
    p_val: float = .05
    "p-value threshold used for significance of the statistical test."
    x_ref_preprocessed: bool = False
    """
    Whether or not the reference data x_ref has already been preprocessed. If True, the reference data will be skipped
    and preprocessing will only be applied to the test data passed to predict.
    """
    preprocess_fn: Optional[Union[Callable, PreprocessConfigResolved]] = None
    "Function to preprocess the data before computing the data drift metrics."
    input_shape: Optional[tuple] = None
    "Optionally pass the shape of the input data. Used when saving detectors."
    data_type: Optional[str] = None
    "Specify data type added to the metadata. E.g. `‘tabular’` or `‘image’`."


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
    sigma: Optional[NDArray[np.float32]] = None
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
    sigma: Optional[NDArray[np.float32]] = None
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
    sigma: Optional[NDArray[np.float32]] = None
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
    sigma: Optional[NDArray[np.float32]] = None
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
    model: Optional[SupportedModels_types] = None
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
}  # type: Dict[str, Type[DriftDetectorConfig]]


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
}  # type: Dict[str, Type[DriftDetectorConfigResolved]]
