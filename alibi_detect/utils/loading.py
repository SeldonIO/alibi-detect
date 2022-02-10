# type: ignore
# TODO: need to rewrite utilities using isinstance or @singledispatch for type checking to work properly
# TODO - Need to modularise torch and tensorflow imports and use. e.g. has_tensorflow and has_pytorch etc
# TODO - further modularisation? e.g. load_kernel_tf and load_kernel_torch? Or refactor to tensorflow/ etc?
# alibi_detect imports
from alibi_detect.version import __version__
from alibi_detect.ad import AdversarialAE, ModelDistillation
from alibi_detect.ad.adversarialae import DenseHidden
from alibi_detect.cd import (ChiSquareDrift, ClassifierDrift, KSDrift, MMDDrift, LSDDDrift, TabularDrift,
                             CVMDrift, FETDrift, SpotTheDiffDrift, ClassifierUncertaintyDrift,
                             RegressorUncertaintyDrift, LearnedKernelDrift)
from alibi_detect.od import (IForest, LLR, Mahalanobis, OutlierAE, OutlierAEGMM, OutlierProphet,
                             OutlierSeq2Seq, OutlierVAE, OutlierVAEGMM, SpectralResidual)
from alibi_detect.od.llr import build_model
# from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_torch
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.utils.tensorflow.kernels import GaussianRBF as GaussianRBF_tf
# from alibi_detect.utils.pytorch.kernels import GaussianRBF as GaussianRBF_torch
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.models.tensorflow import TransformerEmbedding, PixelCNN
from alibi_detect.cd.tensorflow.preprocess import _Encoder
from alibi_detect.models.tensorflow.autoencoder import AE, AEGMM, DecoderLSTM, EncoderLSTM, Seq2Seq, VAE, VAEGMM
from alibi_detect.utils.registry import registry
from alibi_detect.utils.schemas import DETECTOR_CONFIGS, DETECTOR_CONFIGS_RESOLVED, SUPPORTED_MODELS, SupportedModels,\
    __config_spec__
from alibi_detect.utils.tensorflow.kernels import DeepKernel as DeepKernel_tf
# from alibi_detect.utils.pytorch.kernels import DeepKernel as DeepKernel_torch
# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import Model as KerasModel
from tensorflow_probability.python.distributions.distribution import Distribution
# # PyTorch imports  # TODO: pytorch not yet supported.
# import torch
# import torch.nn as nn
# Misc imports
from transformers import AutoTokenizer
import numpy as np
from typing import Union, Optional, Callable, Dict, List, Tuple, Any
from copy import deepcopy
from functools import partial
from pathlib import Path
import dill
import os
import toml
from importlib import import_module
import warnings
import logging


logger = logging.getLogger(__name__)

Data = Union[
    AdversarialAE,
    ChiSquareDrift,
    ClassifierDrift,
    IForest,
    KSDrift,
    LLR,
    Mahalanobis,
    MMDDrift,
    LSDDDrift,
    ModelDistillation,
    OutlierAE,
    OutlierAEGMM,
    OutlierProphet,
    OutlierSeq2Seq,
    OutlierVAE,
    OutlierVAEGMM,
    SpectralResidual,
    TabularDrift,
    CVMDrift,
    FETDrift,
    SpotTheDiffDrift,
    ClassifierUncertaintyDrift,
    RegressorUncertaintyDrift,
    LearnedKernelDrift
]


REQUIRES_BACKEND = [
    ClassifierDrift.__name__,
    ClassifierUncertaintyDrift.__name__,
    LearnedKernelDrift.__name__,
    LSDDDrift.__name__,
    MMDDrift.__name__,
    RegressorUncertaintyDrift.__name__,
    SpotTheDiffDrift.__name__
]

DEFAULT_DETECTORS = [
    'AdversarialAE',
    'IForest',
    'LLR',
    'Mahalanobis',
    'ModelDistillation',
    'OutlierAE',
    'OutlierAEGMM',
    'OutlierProphet',
    'OutlierSeq2Seq',
    'OutlierVAE',
    'OutlierVAEGMM',
    'SpectralResidual',
]

# TODO - add all drift methods in as .get_config() methods are completed
DRIFT_DETECTORS = [  # Drift detectors separated out as they now have their own save methods
    'MMDDrift',
    'LSDDDrift',
    'ChiSquareDrift',
    'TabularDrift',
    'KSDrift',
    'CVMDrift',
    'FETDrift',
    'ClassifierDrift',
    'SpotTheDiffDrift',
    'LearnedKernelDrift'
]

# Fields to resolve in resolve_cfg ("resolve" meaning either load local artefact or resolve @registry, conversion to
# tuple, np.ndarray and np.dtype are dealt with separately).
FIELDS_TO_RESOLVE = [
    ['preprocess_fn'],
    ['preprocess_fn', 'src'],
    ['preprocess_fn', 'model'],
    ['preprocess_fn', 'embedding'],
    ['preprocess_fn', 'tokenizer'],
    ['x_ref'],
    ['model'],
    ['optimizer'],
    ['reg_loss_fn'],
    ['kernel'],
    ['dataset'],
    ['kernel', 'src'],
    ['kernel', 'proj'],
    ['kernel', 'kernel_a', 'src'],
    ['kernel', 'kernel_b', 'src'],
    ['initial_diffs']
]

# Directories to amend before resolving config (fields to prepend config file dir to)
DIR_FIELDS = [
    ['preprocess_fn'],
    ['preprocess_fn', 'src'],
    ['preprocess_fn', 'model', 'src'],
    ['preprocess_fn', 'embedding', 'src'],
    ['preprocess_fn', 'tokenizer', 'src'],
    ['x_ref'],
    ['model', 'src'],
    ['kernel'],
    ['kernel', 'src'],
    ['optimizer'],
    ['reg_loss_fn'],
    ['kernel', 'proj', 'src'],
    ['kernel', 'kernel_a', 'src'],
    ['kernel', 'kernel_b', 'src'],
    ['initial_diffs']
]

# Fields to convert from list to tuple in resolve_cfg
FIELDS_TO_TUPLE = [
    ['detector', 'kwargs', 'input_shape']
]

# Fields to convert from list to np.ndarray in resolve_cfg
FIELDS_TO_ARRAY = [
    ['sigma'],
    ['kernel', 'sigma'],
    ['kernel', 'kernel_a', 'sigma'],
    ['kernel', 'kernel_b', 'sigma'],
]

# Fields to convert from str to np.dtype
FIELDS_TO_DTYPE = [
    ['preprocess_fn', 'dtype']
]


def load_detector(filepath: Union[str, os.PathLike], **kwargs) -> Data:
    """
    Load outlier, drift or adversarial detector.

    Parameters
    ----------
    filepath
        Load directory.

    Returns
    -------
    Loaded outlier or adversarial detector object.
    """
    filepath = Path(filepath)
    # If reference is a 'config.toml' itself, pass to new load function
    if filepath.name == 'config.toml':
        return _load_detector_config(filepath)

    # Otherwise, if a directory, look for meta.dill, meta.pickle or config.toml inside it
    elif filepath.is_dir():
        files = [str(f.name) for f in filepath.iterdir() if f.is_file()]
        if 'config.toml' in files:
            return _load_detector_config(filepath.joinpath('config.toml'))
        elif 'meta.dill' in files:
            return _load_detector_legacy(filepath, '.dill', **kwargs)
        elif 'meta.pickle' in files:
            return _load_detector_legacy(filepath, '.pickle', **kwargs)
        else:
            raise ValueError('Neither meta.dill, meta.pickle or config.toml exist in {}.'.format(filepath))

    # No other file types are accepted, so if not dir raise error
    else:
        raise ValueError("load_detector accepts only a filepath to a directory, or a config.toml file.")


# TODO - will eventually become load_detector
def _load_detector_config(cfg: Union[str, os.PathLike, dict]) -> Data:
    """
    Loads a drift detector specified in a detector config dict. Validation is performed with pydantic.

    Parameters
    ----------
    cfg
        The detector config dict.

    Returns
    -------
    The instantiated detector.
    """
    # Load toml if needed
    if isinstance(cfg, (str, os.PathLike)):
        config_file = Path(deepcopy(cfg))
        config_dir = config_file.parent
        cfg = read_config(config_file)
    elif isinstance(cfg, dict):
        config_file = None
        config_dir = None
    else:
        raise ValueError('Detector `cfg` not recognised.')

    # Resolve and validate config
    cfg = validate_config(cfg)
    logger.info('Validated unresolved config.')
    cfg = resolve_cfg(cfg, config_dir=config_dir)
    print(type(cfg.get('dataset')))
    cfg = validate_config(cfg, resolved=True)
    logger.info('Validated resolved config.')

    # Backend and detector type
    detector_name = cfg.get('name')
    backend = cfg.pop('backend')  # popping so that cfg left as kwargs + `name` when passed to _init_detector
    if backend.lower() != 'tensorflow':
        raise NotImplementedError('Loading detectors with PyTorch or sklearn backend is not yet supported.')

    # Get x_ref
    x_ref = cfg.pop('x_ref')
    if isinstance(x_ref, str):  # If x_ref still a str, resolving must have failed
        raise ValueError("Failed to resolve x_ref field.")

    # Get kernel
    kernel = cfg.pop('kernel', None)  # Don't need to check if None as kernel=None defaults to GaussianRBF
    if isinstance(kernel, dict):
        logger.info('Loading kernel.')
        kernel = _load_kernel(kernel, backend, cfg['device'])

    # Get preprocess_fn
    preprocess_fn = cfg.pop('preprocess_fn')
    if isinstance(preprocess_fn, dict):
        logger.info('Loading preprocess_fn.')
        preprocess_fn = _load_preprocess(preprocess_fn, backend=backend)

    # Get model
    model = cfg.pop('model', None)
    if model is not None:
        if not isinstance(model, SupportedModels):
            raise ValueError(f"Failed to load the {detector_name}'s model."
                             "Is the `model` field specified, and is the model a supported type?")

    # Init detector
    logger.info('Instantiating detector.')
    detector = _init_detector(x_ref, cfg, preprocess_fn=preprocess_fn, model=model, kernel=kernel, backend=backend)

    # Update metadata
    detector.meta.update({'config_file': str(config_file.resolve())})

    logger.info('Finished loading detector.')
    return detector


def validate_config(cfg: dict, resolved: bool = False) -> dict:
    """
    Validates a detector config dict by passing the dict to the detector's pydantic model schema.

    Parameters
    ----------
    cfg
        The detector config dict.
    resolved
        Whether the config is resolved or not. For example, if resolved=True, `x_ref` is expected to be a
        np.ndarray, wheras if resolved=False, `x_ref` is expected to be a str.

    Returns
    -------
    The validated config dict, with missing fields set to their default values.
    """
    # Get detector name
    if 'name' in cfg:
        detector_name = cfg['name']
    else:
        raise ValueError('`name` missing from config.toml.')

    # Validate detector specific config
    if detector_name in DETECTOR_CONFIGS.keys():
        if resolved:
            cfg = DETECTOR_CONFIGS_RESOLVED[detector_name](**cfg).dict()  # type: ignore[attr-defined]
        else:
            cfg = DETECTOR_CONFIGS[detector_name](**cfg).dict()  # type: ignore[attr-defined]
    else:
        raise ValueError('Loading the specified detector from a config.toml is not yet supported.')

    # check version
    version = cfg.pop('version', None)
    if version is not None and version != __version__:
        warnings.warn(f'Trying to load detector from version {version} when using version '
                      f'{__version__}. This may lead to breaking code or invalid results.')

    # Check config specification version
    config_spec = cfg.pop('config_spec', None)
    if config_spec is not None and config_spec != __config_spec__:
        warnings.warn(f'Trying to load detector from  config with specification {version} when the installed '
                      f'alibi-detect version expects specification {__config_spec__}.'
                      'This may lead to breaking code or invalid results.')
    return cfg


def _init_detector(x_ref: Union[np.ndarray, list],
                   cfg: dict,
                   preprocess_fn: Optional[Callable] = None,
                   model: Optional[SUPPORTED_MODELS] = None,
                   kernel: Optional[Callable] = None,
                   backend: Optional[str] = 'tensorflow') -> Data:
    """
    Instantiates a detector (x_ref, preprocess_fn, model, etc in the dict should be fully
    resolved runtime objects).

    Parameters
    ----------
    x_ref
        The reference data.
    cfg
        The detectors config dict (with x_ref, model etc already pop'ed from it, such that what remains are the kwargs).
    preprocess_fn
        Optional preprocessing function.
    model
        Optional model (e.g. for ClassifierDrift).
    kernel
        Optional kernel (e.g. for MMDDrift).
    backend
        The backend.

    Returns
    -------
    The instantiated detector.
    """
    detector_name = cfg.pop('name', None)

    # Process args
    args = [x_ref]
    if model is not None:
        args.append(model)

    # Process kwargs (cfg should just be kwargs by this point)
    if detector_name in REQUIRES_BACKEND:
        cfg.update({'backend': backend})
    if preprocess_fn is not None:
        cfg.update({'preprocess_fn': preprocess_fn})
    if kernel is not None:
        cfg.update({'kernel': kernel})

    # Instantiate the detector
    klass = getattr(import_module('alibi_detect.cd'), detector_name)
    detector = klass(*args, **cfg)
    logger.info('Instantiated drift detector %s', detector_name)
    return detector


def _load_kernel(cfg: dict, backend: str = 'tensorflow', device: Optional[str] = None) -> Callable:
    """
    Loads a kernel from a kernel config dict.

    Parameters
    ----------
    cfg
        A kernel config dict. (see pydantic schema's).
    backend
        The backend.
    device
        The device (pytorch backend only).

    Returns
    -------
    The kernel.
    """

    if 'src' in cfg:  # Standard kernel config
        kernel = cfg['src']
        sigma = cfg['sigma']
        if callable(kernel):
            if kernel == GaussianRBF_tf:
                sigma = tf.convert_to_tensor(sigma) if isinstance(sigma, np.ndarray) else sigma
                kernel = kernel(sigma=sigma, trainable=cfg['trainable'])
#            elif kernel == GaussianRBF_torch:  # TODO
#                raise NotImplementedError('Loading PyTorch kernels not currently supported.')
#                torch_device = _set_device(device)
#                sigma = torch.from_numpy(sigma).to(torch_device) if isinstance(sigma, np.ndarray) else None
#                kernel = kernel(sigma=sigma, trainable=cfg['trainable'])
            else:
                kwargs = cfg['kwargs']
                kernel = kernel(**kwargs)

    elif 'proj' in cfg:  # DeepKernel config
        proj = cfg['proj']
        eps = cfg['eps']
        # Kernel a
        kernel_a = cfg['kernel_a']
        if kernel_a is not None:
            kernel_a = _load_kernel(kernel_a, backend, device)
        else:
            kernel_a = GaussianRBF_tf(trainable=True) if backend == 'tensorflow' else None
            # GaussianRBF_torch(trainable=True)
        # Kernel b
        kernel_b = cfg['kernel_b']
        if kernel_b is not None:
            kernel_b = _load_kernel(kernel_b, backend, device)
        else:
            kernel_b = GaussianRBF_tf(trainable=True) if backend == 'tensorflow' else None
            # GaussianRBF_torch(trainable=True)

        # Assemble deep kernel
        if backend == 'tensorflow':
            kernel = DeepKernel_tf(proj, kernel_a=kernel_a, kernel_b=kernel_b, eps=eps)
        else:
            kernel = None  # DeepKernel_torch(proj, kernel_a=kernel_a, kernel_b=kernel_b, eps=eps)
    else:
        raise ValueError('Unable to process kernel.)')
    return kernel


def _load_preprocess(cfg: dict,
                     backend: Optional[str] = 'tensorflow') -> Optional[Callable]:
    """
    This function builds a preprocess_fn from the preprocess dict in a detector config dict. The dict format is
    expected to match that generated by serialize_preprocess in alibi_detect.utils.saving (also see pydantic schema).
    The model, tokenizer and preprocess_batch_fn are expected to be already resolved.

    Parameters
    ----------
    cfg
        A preprocess_fn config dict. (see pydantic schemas).
    backend
        The backend.

    Returns
    -------
    The preprocess_fn function.
    """
    preprocess_fn = cfg.pop('src')

    if callable(preprocess_fn):
        if preprocess_fn == preprocess_drift_tf:  # or preprocess_fn == preprocess_drift_torch:
            # If preprocess_drift function, kwargs is preprocess cfg minus 'function' and 'kwargs'
            cfg.pop('kwargs')
            kwargs = cfg.copy()

            # Final processing of model (and/or embedding)
            model = kwargs['model']
            emb = kwargs.pop('embedding')  # embedding passed to preprocess_drift as `model` therefore remove
            model = _prep_model_and_embedding(model, emb, backend=backend)
            if model is None:
                raise ValueError("A 'model'  and/or `embedding` must be specified when "
                                 "preprocess_fn='preprocess_drift'")
            kwargs.update({'model': model})

            # Backend specifics
            if backend == 'tensorflow':
                assert preprocess_fn == preprocess_drift_tf
                if not isinstance(model, KerasModel):
                    raise ValueError('The specified model is not a compatible tensorflow model.')
                kwargs.pop('device')
            elif backend == 'pytorch':  # TODO
                raise NotImplementedError('Loading preprocess_fn for PyTorch not yet supported.')
#                assert preprocess_fn == preprocess_drift_torch
#                if not isinstance(model, (nn.Module, nn.Sequential)):
#                    raise ValueError('The specified model is not a compatible pytorch model.')
#                device = cfg['device']
#                if device is not None:
#                    device = torch.device('cuda' if torch.cuda.is_available() else device)
#                    kwargs.update({'model': kwargs['model'].to(device)})
#                    kwargs.update({'device': device})
        else:
            kwargs = cfg['kwargs']  # If generic callable, kwargs is cfg['kwargs']

    else:
        logger.warning('Unable to process preprocess_fn. No preprocessing function is defined.')
        return None

    return partial(preprocess_fn, **kwargs)


def _load_model(cfg: dict,
                backend: str) -> SUPPORTED_MODELS:
    """
    Loads TensorFlow, PyTorch and scikit-learn models (currently only TensorFlow supported), from a model config
    dict.

    Note: Handling of different model types ('UAE', 'HiddenOutput' and 'custom' are TBC).

    Parameters
    ----------
    cfg
        Model config dict. (see pydantic model schemas).
    backend
        The backend.

    Returns
    -------
    The loaded model.
    """

    # Load model
    src = cfg['src']
    typ = cfg['type']
    custom_obj = cfg['custom_obj']
    if src is None:
        raise ValueError("No 'src' field for 'model'.")
    src = Path(src)
    if not src.is_dir():
        raise FileNotFoundError("Compatible model not found at %s" % str(src.resolve()))

    if backend == 'tensorflow':
        model = load_tf_model(src, load_dir='.', custom_objects=custom_obj)
        if typ == 'UAE':
            model = UAE(encoder_net=model)
        elif typ == 'HiddenOutput' or typ == 'custom':
            pass
        else:
            raise ValueError("Model 'type' not recognised.")
    else:
        raise NotImplementedError('Loading of pytorch models not currently supported')

    return model


def _load_embedding(cfg: dict) -> TransformerEmbedding:
    """
    Load a pre-trained text embedding from an embedding config dict.

    Parameters
    ----------
    cfg
    An embedding config dict. (see the pydantic schemas).

    Returns
    -------
    The loaded embedding.
    """
    src = cfg['src']
    layers = cfg['layers']
    typ = cfg['type']
    if src is None:
        raise ValueError("No 'src' field for `embedding'")
    src = Path(src)
    if not src.is_dir():
        raise FileNotFoundError("Compatible embedding not found at %s" % str(src.resolve()))

    emb = TransformerEmbedding(str(src), embedding_type=typ, layers=layers)
    return emb


def _load_tokenizer(cfg: dict) -> AutoTokenizer:
    """
    Loads a text tokenizer from a tokenizer config dict.

    Parameters
    ----------
    cfg
        A tokenizer config dict. (see the pydantic schemas).

    Returns
    -------
    The loaded tokenizer.
    """
    src = cfg['src']
    kwargs = cfg['kwargs']
    if src is None:
        raise ValueError("No 'src' field for 'tokenizer'")
    src = Path(src)
    if not src.is_dir():
        raise FileNotFoundError("Compatible tokenizer not found at %s" % str(src.resolve()))

    tokenizer = AutoTokenizer.from_pretrained(src, **kwargs)
    return tokenizer


def _load_optimizer(cfg: dict,
                    backend: str) -> Union[tf.keras.optimizers.Optimizer, Callable]:
    """
    Loads an optimzier from an optimizer config dict. When backend='tensorflow', the config dict should be in
    the format given by tf.keras.optimizers.serialize().

    Parameters
    ----------
    cfg
        The optimizer config dict.
    backend
        The backend.

    Returns
    -------
    The loaded optimizer.
    """
    if backend == 'tensorflow':
        optimizer = tf.keras.optimizers.deserialize(cfg)
    else:
        raise NotImplementedError('Loading of pytorch optimizers not currently supported')

    return optimizer


def _prep_model_and_embedding(model: Optional[SUPPORTED_MODELS], emb: Optional[TransformerEmbedding],
                              backend: str) -> SUPPORTED_MODELS:
    """
    Function to perform final preprocessing of model before it is passed to preprocess_drift. This is separated from
    load_model in order to reduce complexity of load functions (with future model load functionality in mind), and also
    to separate embedding logic from model loading (allows for cleaner config layout and resolution of it).

    Parameters
    ----------
    model
        A compatible model.
    emb
        A text embedding model.
    backend
        The detector backend (backend in this case actually determines tensorflow vs pytorch model).

    Returns
    -------
    The final model ready to passed to preprocess_drift.
    """
    # If a model exists, process it
    if model is not None:
        if backend == 'tensorflow':
            model = model.encoder if isinstance(model, UAE) else model  # This is to avoid nesting UAE's
            if emb is not None:
                model = _Encoder(emb, mlp=model)
            model = UAE(encoder_net=model)  # TODO - do we want to wrap model in UAE? Not always necessary...

        else:
            raise NotImplementedError("Loading of pytorch models is not currently implemented.")

    # If no model exists, store embedding in model (both may be None)
    else:
        model = emb

    return model


def _get_nested_value(dic: dict, keys: list) -> Any:
    """
    Get a value from a nested dictionary.

    Parameters
    ----------
    dic
        The dictionary.
    keys
        List of keys to "walk" to nested value.
        For example, to extract the value `dic['key1']['key2']['key3']`, set `keys = ['key1', 'key2', 'key3']`.

    Returns
    -------
    The nested value specified by `keys`.
    """
    for key in keys:
        try:
            dic = dic[key]
        except (TypeError, KeyError):
            return None
    return dic


def _set_nested_value(dic: dict, keys: list, value: Any):
    """
    Set a value in a nested dictionary.

    Parameters
    ----------
    dic
        The dictionary.
    keys
        List of keys to "walk" to nested value.
        For example, to set the value `dic['key1']['key2']['key3']`, set `keys = ['key1', 'key2', 'key3']`.
    value
        The value to set.
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def read_config(filepath: Union[os.PathLike, str]) -> dict:
    """
    This function reads a detector toml config file and returns a dict specifying the detector.

    Parameters
    ----------
    filepath
        The filepath to the config.toml file.

    Returns
    -------
        Parsed toml dictionary.
    """
    filepath = Path(filepath)
    cfg = toml.load(filepath)
    logger.info('Loaded config file from %s', str(filepath))

    # Convert keys in categories_per_feature back to str. This is undesirable, but currently necessary in order
    # to use toml library, as error with toml dumping dict with int keys.
    if 'categories_per_feature' in cfg:
        new = {}
        for key in cfg['categories_per_feature']:
            new[int(key)] = cfg['categories_per_feature'][key]
        cfg['categories_per_feature'] = new

    # This is necessary as no None/null in toml spec., and missing values are set to defaults set in pydantic models.
    # But we sometimes to do explicitly spec as None.
    cfg = _replace(cfg, "None", None)

    return cfg  # type: ignore[return-value] # TODO - toml actually returns MutableMapping, consider updating throughout


def resolve_cfg(cfg: dict, config_dir: Optional[Path]) -> dict:
    """
    Resolves artefacts in a config dict. For example x_ref='x_ref.npy' is resolved by loading the np.ndarray from
    the .npy file. For a list of fields that are resolved, see
    https://docs.seldon.io/projects/alibi-detect/en/latest/overview/config_file.html.

    Parameters
    ----------
    cfg
        The unresolved config dict.
    config_dir
        Filepath to directory the `config.toml` is located in. Only required if different from the
        runtime directory, and artefacts are specified with filepaths relative to the config.toml file.

    Returns
    -------
    The resolved config dict.
    """
    # Before main resolution, update filepaths relative to config file
    if config_dir is not None:
        for key in DIR_FIELDS:
            src = _get_nested_value(cfg, key)
            if isinstance(src, str):
                src = config_dir.joinpath(Path(src))
                if src.is_file() or src.is_dir():
                    _set_nested_value(cfg, key, str(src))

    # Resolve filepaths (load files) and resolve function/object registries
    for key in FIELDS_TO_RESOLVE:
        logger.info('Resolving config field: {}.'.format(key))

        src = _get_nested_value(cfg, key)

        obj = None
        # Resolve runtime registered function/object
        if isinstance(src, str):
            if src.startswith('@'):
                src = src[1:]
                if src in registry.get_all():
                    obj = registry.get(src)
                else:
                    raise ValueError("Can't find %s in the custom function registry" % src)
                logger.info('Successfully resolved registry entry %s' % src)

            # Load dill or numpy file
            elif Path(src).is_file():
                if Path(src).suffix == '.dill':
                    obj = dill.load(open(src, 'rb'))
                if Path(src).suffix == '.npy':
                    obj = np.load(src)

#            # Pytorch device  # TODO
#            elif key[-1] == 'device':
#                obj = _set_device(src)

        # Resolve dict spec
        elif isinstance(src, dict):
            backend = cfg.get('backend', 'tensorflow')
            if key[-1] in ('model', 'proj'):
                obj = _load_model(src, backend=backend)
            if key[-1] == 'embedding':
                obj = _load_embedding(src)
            elif key[-1] == 'tokenizer':
                obj = _load_tokenizer(src)
            elif key[-1] == 'optimizer':
                obj = _load_optimizer(src, backend=backend)

        # Put the resolved function into the cfg dict
        if obj is not None:
            _set_nested_value(cfg, key, obj)

    # Convert selected lists to tuples
    for key in FIELDS_TO_TUPLE:
        val = _get_nested_value(cfg, key)
        if val is not None:
            _set_nested_value(cfg, key, tuple(val))

    # Convert selected lists to np.ndarray's
    for key in FIELDS_TO_ARRAY:
        val = _get_nested_value(cfg, key)
        if val is not None:
            _set_nested_value(cfg, key, np.array(val))

    # Convert selected str's to np.dtype's
    for key in FIELDS_TO_DTYPE:
        val = _get_nested_value(cfg, key)
        if val is not None:
            val = val.split('.')
#            val[0] = np if val[0] == 'np' else tf if val[0] == 'tf' else torch if val[0] == 'torch' else None  TODO
            val[0] = np if val[0] == 'np' else tf if val[0] == 'tf' else None
            if val[0] is None:
                raise ValueError("`dtype` must be in format np.<dtype>, tf.<dtype> or torch.<dtype>.")
            _set_nested_value(cfg, key, getattr(val[0], val[1]))

    return cfg


# def _set_device(device: Optional[str]) -> torch.device:  # TODO
#    """
#    Set PyTorch device.
#
#    Parameters
#    ----------
#    device
#        String identifying the device.
#
#    Returns
#    -------
#    A set torch.device object.
#    """
#    if device is None or device in ['gpu', 'cuda']:
#        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#        if torch_device.type == 'cpu':
#            logger.warning('No GPU detected, fall back on CPU.')
#    else:
#        torch_device = torch.device('cpu')
#    return torch_device


def _load_detector_legacy(filepath: Union[str, os.PathLike], suffix: str, **kwargs) -> Data:
    """
    Legacy function to load outlier, drift or adversarial detectors stored dill or pickle files.

    Warning
    -------
    This function will be removed in a future version.

    Parameters
    ----------
    filepath
        Load directory.
    suffix
        File suffix for meta and state files. Either `'.dill'` or `'.pickle'`.

    Returns
    -------
    Loaded outlier or adversarial detector object.
    """
    warnings.warn(' Loading of meta.dill and meta.pickle files will be removed in a future version.',
                  DeprecationWarning, 3)

    if kwargs:
        k = list(kwargs.keys())
    else:
        k = []

    # check if path exists
    filepath = Path(filepath)
    if not filepath.is_dir():
        raise ValueError('{} does not exist.'.format(filepath))

    # load metadata
    meta_dict = dill.load(open(filepath.joinpath('meta' + suffix), 'rb'))

    # check version
    try:
        if meta_dict['version'] != __version__:
            warnings.warn(f'Trying to load detector from version {meta_dict["version"]} when using version '
                          f'{__version__}. This may lead to breaking code or invalid results.')
    except KeyError:
        warnings.warn('Trying to load detector from an older version.'
                      'This may lead to breaking code or invalid results.')

    if 'backend' in list(meta_dict.keys()) and meta_dict['backend'] == 'pytorch':
        raise NotImplementedError('Detectors with PyTorch backend are not yet supported.')

    detector_name = meta_dict['name']
    if detector_name not in DEFAULT_DETECTORS and detector_name not in DRIFT_DETECTORS:
        raise ValueError('{} is not supported by `load_detector`.'.format(detector_name))

    # load outlier detector specific parameters
    state_dict = dill.load(open(filepath.joinpath(detector_name + suffix), 'rb'))

    # initialize outlier detector
    if detector_name == 'OutlierAE':
        ae = load_tf_ae(filepath)
        detector = _init_od_ae(state_dict, ae)
    elif detector_name == 'OutlierVAE':
        vae = load_tf_vae(filepath, state_dict)
        detector = _init_od_vae(state_dict, vae)
    elif detector_name == 'Mahalanobis':
        detector = _init_od_mahalanobis(state_dict)
    elif detector_name == 'IForest':
        detector = _init_od_iforest(state_dict)
    elif detector_name == 'OutlierAEGMM':
        aegmm = load_tf_aegmm(filepath, state_dict)
        detector = _init_od_aegmm(state_dict, aegmm)
    elif detector_name == 'OutlierVAEGMM':
        vaegmm = load_tf_vaegmm(filepath, state_dict)
        detector = _init_od_vaegmm(state_dict, vaegmm)
    elif detector_name == 'AdversarialAE':
        ae = load_tf_ae(filepath)
        custom_objects = kwargs['custom_objects'] if 'custom_objects' in k else None
        model = load_tf_model(filepath, custom_objects=custom_objects)
        model_hl = load_tf_hl(filepath, model, state_dict)
        detector = _init_ad_ae(state_dict, ae, model, model_hl)
    elif detector_name == 'ModelDistillation':
        md = load_tf_model(filepath, load_dir='distilled_model')
        custom_objects = kwargs['custom_objects'] if 'custom_objects' in k else None
        model = load_tf_model(filepath, custom_objects=custom_objects)
        detector = _init_ad_md(state_dict, md, model)
    elif detector_name == 'OutlierProphet':
        detector = _init_od_prophet(state_dict)
    elif detector_name == 'SpectralResidual':
        detector = _init_od_sr(state_dict)
    elif detector_name == 'OutlierSeq2Seq':
        seq2seq = load_tf_s2s(filepath, state_dict)
        detector = _init_od_s2s(state_dict, seq2seq)
    elif detector_name == 'LLR':
        models = load_tf_llr(filepath, **kwargs)
        detector = _init_od_llr(state_dict, models)
    else:
        raise NotImplementedError

    detector.meta = meta_dict
    logger.info('Finished loading detector.')
    return detector


def load_tf_hl(filepath: Union[str, os.PathLike], model: tf.keras.Model, state_dict: dict) -> List[tf.keras.Model]:
    """
    Load hidden layer models for AdversarialAE.

    Parameters
    ----------
    filepath
        Saved model directory.
    model
        tf.keras classification model.
    state_dict
        Dictionary containing the detector's parameters.

    Returns
    -------
    List with loaded tf.keras models.
    """
    model_dir = Path(filepath).joinpath('model')
    hidden_layer_kld = state_dict['hidden_layer_kld']
    if not hidden_layer_kld:
        return []
    model_hl = []
    for i, (hidden_layer, output_dim) in enumerate(hidden_layer_kld.items()):
        m = DenseHidden(model, hidden_layer, output_dim)
        m.load_weights(model_dir.joinpath('model_hl_' + str(i) + '.ckpt'))
        model_hl.append(m)
    return model_hl


def load_tf_ae(filepath: Union[str, os.PathLike]) -> tf.keras.Model:
    """
    Load AE.

    Parameters
    ----------
    filepath
        Saved model directory.

    Returns
    -------
    Loaded AE.
    """
    model_dir = Path(filepath).joinpath('model')
    if not [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No encoder, decoder or ae found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(model_dir.joinpath('encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(model_dir.joinpath('decoder_net.h5'))
    ae = AE(encoder_net, decoder_net)
    ae.load_weights(model_dir.joinpath('ae.ckpt'))
    return ae


def load_tf_vae(filepath: Union[str, os.PathLike],
                state_dict: Dict) -> tf.keras.Model:
    """
    Load VAE.

    Parameters
    ----------
    filepath
        Saved model directory.
    state_dict
        Dictionary containing the latent dimension and beta parameters.

    Returns
    -------
    Loaded VAE.
    """
    model_dir = Path(filepath).joinpath('model')
    if not [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No encoder, decoder or vae found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(model_dir.joinpath('encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(model_dir.joinpath('decoder_net.h5'))
    vae = VAE(encoder_net, decoder_net, state_dict['latent_dim'], beta=state_dict['beta'])
    vae.load_weights(model_dir.joinpath('vae.ckpt'))
    return vae


def load_tf_aegmm(filepath: Union[str, os.PathLike],
                  state_dict: Dict) -> tf.keras.Model:
    """
    Load AEGMM.

    Parameters
    ----------
    filepath
        Saved model directory.
    state_dict
        Dictionary containing the `n_gmm` and `recon_features` parameters.

    Returns
    -------
    Loaded AEGMM.
    """
    model_dir = Path(filepath).joinpath('model')

    if not [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No encoder, decoder, gmm density net or aegmm found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(model_dir.joinpath('encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(model_dir.joinpath('decoder_net.h5'))
    gmm_density_net = tf.keras.models.load_model(model_dir.joinpath('gmm_density_net.h5'))
    aegmm = AEGMM(encoder_net, decoder_net, gmm_density_net, state_dict['n_gmm'], state_dict['recon_features'])
    aegmm.load_weights(model_dir.joinpath('aegmm.ckpt'))
    return aegmm


def load_tf_vaegmm(filepath: Union[str, os.PathLike],
                   state_dict: Dict) -> tf.keras.Model:
    """
    Load VAEGMM.

    Parameters
    ----------
    filepath
        Saved model directory.
    state_dict
        Dictionary containing the `n_gmm`, `latent_dim` and `recon_features` parameters.

    Returns
    -------
    Loaded VAEGMM.
    """
    model_dir = Path(filepath).joinpath('model')
    if not [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No encoder, decoder, gmm density net or vaegmm found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(model_dir.joinpath('encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(model_dir.joinpath('decoder_net.h5'))
    gmm_density_net = tf.keras.models.load_model(model_dir.joinpath('gmm_density_net.h5'))
    vaegmm = VAEGMM(encoder_net, decoder_net, gmm_density_net, state_dict['n_gmm'],
                    state_dict['latent_dim'], state_dict['recon_features'], state_dict['beta'])
    vaegmm.load_weights(model_dir.joinpath('vaegmm.ckpt'))
    return vaegmm


def load_tf_s2s(filepath: Union[str, os.PathLike],
                state_dict: Dict) -> tf.keras.Model:
    """
    Load seq2seq TensorFlow model.

    Parameters
    ----------
    filepath
        Saved model directory.
    state_dict
        Dictionary containing the `latent_dim`, `shape`, `output_activation` and `beta` parameters.

    Returns
    -------
    Loaded seq2seq model.
    """
    model_dir = Path(filepath).joinpath('model')
    if not [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No seq2seq or threshold estimation net found in {}.'.format(model_dir))
        return None
    # load threshold estimator net, initialize encoder and decoder and load seq2seq weights
    threshold_net = tf.keras.models.load_model(model_dir.joinpath('threshold_net.h5'), compile=False)
    latent_dim = state_dict['latent_dim']
    n_features = state_dict['shape'][-1]
    encoder_net = EncoderLSTM(latent_dim)
    decoder_net = DecoderLSTM(latent_dim, n_features, state_dict['output_activation'])
    seq2seq = Seq2Seq(encoder_net, decoder_net, threshold_net, n_features, beta=state_dict['beta'])
    seq2seq.load_weights(model_dir.joinpath('seq2seq.ckpt'))
    return seq2seq


def load_tf_llr(filepath: Union[str, os.PathLike], dist_s: Union[Distribution, PixelCNN] = None,
                dist_b: Union[Distribution, PixelCNN] = None, input_shape: tuple = None):
    """
    Load LLR TensorFlow models or distributions.

    Parameters
    ----------
    detector
        Likelihood ratio detector.
    filepath
        Saved model directory.
    dist_s
        TensorFlow distribution for semantic model.
    dist_b
        TensorFlow distribution for background model.
    input_shape
        Input shape of the model.

    Returns
    -------
    Detector with loaded models.
    """
    model_dir = Path(filepath).joinpath('model')
    h5files = [f.name for f in model_dir.glob('[!.]*.h5')]
    if 'model_s.h5' in h5files and 'model_b.h5' in h5files:
        model_s, dist_s = build_model(dist_s, input_shape, str(model_dir.joinpath('model_s.h5').resolve()))
        model_b, dist_b = build_model(dist_b, input_shape, str(model_dir.joinpath('model_b.h5').resolve()))
        return dist_s, dist_b, model_s, model_b
    else:
        dist_s = tf.keras.models.load_model(model_dir.joinpath('model.h5'), compile=False)
        if 'model_background.h5' in h5files:
            dist_b = tf.keras.models.load_model(model_dir.joinpath('model_background.h5'), compile=False)
        else:
            dist_b = None
        return dist_s, dist_b, None, None


def _init_od_ae(state_dict: Dict,
                ae: tf.keras.Model) -> OutlierAE:
    """
    Initialize OutlierVAE.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    ae
        Loaded AE.

    Returns
    -------
    Initialized OutlierAE instance.
    """
    od = OutlierAE(threshold=state_dict['threshold'], ae=ae)
    return od


def _init_od_vae(state_dict: Dict,
                 vae: tf.keras.Model) -> OutlierVAE:
    """
    Initialize OutlierVAE.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    vae
        Loaded VAE.

    Returns
    -------
    Initialized OutlierVAE instance.
    """
    od = OutlierVAE(threshold=state_dict['threshold'],
                    score_type=state_dict['score_type'],
                    vae=vae,
                    samples=state_dict['samples'])
    return od


def _init_ad_ae(state_dict: Dict,
                ae: tf.keras.Model,
                model: tf.keras.Model,
                model_hl: List[tf.keras.Model]) -> AdversarialAE:
    """
    Initialize AdversarialAE.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    ae
        Loaded VAE.
    model
        Loaded classification model.
    model_hl
        List of tf.keras models.

    Returns
    -------
    Initialized AdversarialAE instance.
    """
    ad = AdversarialAE(threshold=state_dict['threshold'],
                       ae=ae,
                       model=model,
                       model_hl=model_hl,
                       w_model_hl=state_dict['w_model_hl'],
                       temperature=state_dict['temperature'])
    return ad


def _init_ad_md(state_dict: Dict,
                distilled_model: tf.keras.Model,
                model: tf.keras.Model) -> ModelDistillation:
    """
    Initialize ModelDistillation.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    distilled_model
        Loaded distilled model.
    model
        Loaded classification model.

    Returns
    -------
    Initialized ModelDistillation instance.
    """
    ad = ModelDistillation(threshold=state_dict['threshold'],
                           distilled_model=distilled_model,
                           model=model,
                           temperature=state_dict['temperature'],
                           loss_type=state_dict['loss_type'])
    return ad


def _init_od_aegmm(state_dict: Dict,
                   aegmm: tf.keras.Model) -> OutlierAEGMM:
    """
    Initialize OutlierAEGMM.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    aegmm
        Loaded AEGMM.

    Returns
    -------
    Initialized OutlierAEGMM instance.
    """
    od = OutlierAEGMM(threshold=state_dict['threshold'],
                      aegmm=aegmm)
    od.phi = state_dict['phi']
    od.mu = state_dict['mu']
    od.cov = state_dict['cov']
    od.L = state_dict['L']
    od.log_det_cov = state_dict['log_det_cov']

    if not all(tf.is_tensor(_) for _ in [od.phi, od.mu, od.cov, od.L, od.log_det_cov]):
        logger.warning('Loaded AEGMM detector has not been fit.')

    return od


def _init_od_vaegmm(state_dict: Dict,
                    vaegmm: tf.keras.Model) -> OutlierVAEGMM:
    """
    Initialize OutlierVAEGMM.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    vaegmm
        Loaded VAEGMM.

    Returns
    -------
    Initialized OutlierVAEGMM instance.
    """
    od = OutlierVAEGMM(threshold=state_dict['threshold'],
                       vaegmm=vaegmm,
                       samples=state_dict['samples'])
    od.phi = state_dict['phi']
    od.mu = state_dict['mu']
    od.cov = state_dict['cov']
    od.L = state_dict['L']
    od.log_det_cov = state_dict['log_det_cov']

    if not all(tf.is_tensor(_) for _ in [od.phi, od.mu, od.cov, od.L, od.log_det_cov]):
        logger.warning('Loaded VAEGMM detector has not been fit.')

    return od


def _init_od_s2s(state_dict: Dict,
                 seq2seq: tf.keras.Model) -> OutlierSeq2Seq:
    """
    Initialize OutlierSeq2Seq.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    seq2seq
        Loaded seq2seq model.

    Returns
    -------
    Initialized OutlierSeq2Seq instance.
    """
    seq_len, n_features = state_dict['shape'][1:]
    od = OutlierSeq2Seq(n_features,
                        seq_len,
                        threshold=state_dict['threshold'],
                        seq2seq=seq2seq,
                        latent_dim=state_dict['latent_dim'],
                        output_activation=state_dict['output_activation'])
    return od


def _load_text_embed(filepath: Union[str, os.PathLike], load_dir: str = 'model') \
        -> Tuple[TransformerEmbedding, Callable]:
    model_dir = Path(filepath).joinpath(load_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir.resolve()))
    args = dill.load(open(model_dir.joinpath('embedding.dill'), 'rb'))
    emb = TransformerEmbedding(
        str(model_dir.resolve()), embedding_type=args['embedding_type'], layers=args['layers']
    )
    return emb, tokenizer


def _init_od_mahalanobis(state_dict: Dict) -> Mahalanobis:
    """
    Initialize Mahalanobis.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.

    Returns
    -------
    Initialized Mahalanobis instance.
    """
    od = Mahalanobis(threshold=state_dict['threshold'],
                     n_components=state_dict['n_components'],
                     std_clip=state_dict['std_clip'],
                     start_clip=state_dict['start_clip'],
                     max_n=state_dict['max_n'],
                     cat_vars=state_dict['cat_vars'],
                     ohe=state_dict['ohe'])
    od.d_abs = state_dict['d_abs']
    od.clip = state_dict['clip']
    od.mean = state_dict['mean']
    od.C = state_dict['C']
    od.n = state_dict['n']
    return od


def _init_od_iforest(state_dict: Dict) -> IForest:
    """
    Initialize isolation forest.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.

    Returns
    -------
    Initialized IForest instance.
    """
    od = IForest(threshold=state_dict['threshold'])
    od.isolationforest = state_dict['isolationforest']
    return od


def _init_od_prophet(state_dict: Dict) -> OutlierProphet:
    """
    Initialize OutlierProphet.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.

    Returns
    -------
    Initialized OutlierProphet instance.
    """
    od = OutlierProphet(cap=state_dict['cap'])
    od.model = state_dict['model']
    return od


def _init_od_sr(state_dict: Dict) -> SpectralResidual:
    """
    Initialize spectral residual detector.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.

    Returns
    -------
    Initialized SpectralResidual instance.
    """
    od = SpectralResidual(threshold=state_dict['threshold'],
                          window_amp=state_dict['window_amp'],
                          window_local=state_dict['window_local'],
                          n_est_points=state_dict['n_est_points'],
                          n_grad_points=state_dict['n_grad_points'])
    return od


def _init_od_llr(state_dict: Dict, models: tuple) -> LLR:
    """
    Initialize LLR detector.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.

    Returns
    -------
    Initialized LLR instance.
    """
    od = LLR(threshold=state_dict['threshold'],
             model=models[0],
             model_background=models[1],
             log_prob=state_dict['log_prob'],
             sequential=state_dict['sequential'])
    if models[2] is not None and models[3] is not None:
        od.model_s = models[2]
        od.model_b = models[3]
    return od


def load_tf_model(filepath: Union[str, os.PathLike],
                  load_dir: str = 'model',
                  custom_objects: dict = None,
                  model_name: str = 'model') -> Optional[tf.keras.Model]:
    """
    Load TensorFlow model.

    Parameters
    ----------
    filepath
        Saved model directory.
    load_dir
        Name of saved model folder within the filepath directory.
    custom_objects
        Optional custom objects when loading the TensorFlow model.
    model_name
        Name of loaded model.

    Returns
    -------
    Loaded model.
    """
    # TODO - update to accept tf format - later PR? (like in save_tf_model, remove model_name for this)
    model_dir = Path(filepath).joinpath(load_dir)
    # Check if path exists
    if not model_dir.is_dir():
        logger.warning('Directory {} does not exist.'.format(model_dir))
        return None
    # Check if model exists
    if model_name + '.h5' not in [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No model found in {}.'.format(model_dir))
        return None
    model = tf.keras.models.load_model(model_dir.joinpath(model_name + '.h5'), custom_objects=custom_objects)
    return model


def _replace(cfg: dict, orig: Optional[str], new: Optional[str]) -> dict:
    """
    Recursively traverse a nested dictionary and replace values.

    Parameters
    ----------
    cfg
        The dictionary.
    orig
        Original value to search.
    new
        Value to replace original with.

    Returns
    -------
    The updated dictionary.
    """
    for k, v in cfg.items():
        if v == orig:
            cfg[k] = new
        elif isinstance(v, dict):
            _replace(v, orig, new)
    return cfg
