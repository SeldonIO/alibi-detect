from alibi_detect.version import __version__
from alibi_detect.cd import ChiSquareDrift, KSDrift, MMDDrift, TabularDrift, LSDDDrift
from alibi_detect.cd import ClassifierUncertaintyDrift, RegressorUncertaintyDrift
from alibi_detect.cd import ClassifierDrift, LearnedKernelDrift, SpotTheDiffDrift
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_torch
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.utils.tensorflow.kernels import GaussianRBF as GaussianRBF_tf
from alibi_detect.utils.pytorch.kernels import GaussianRBF as GaussianRBF_torch
from alibi_detect.cd.tensorflow import HiddenOutput, UAE
from alibi_detect.cd.tensorflow.preprocess import _Encoder
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.utils.registry import registry
from alibi_detect.utils.config import DETECTOR_CONFIGS, DETECTOR_CONFIGS_RESOLVED, SUPPORTED_MODELS, __config_spec__
from alibi_detect.utils.tensorflow.kernels import DeepKernel
import numpy as np
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import Model as KerasModel
from typing import Union, Optional, Callable
import logging
from copy import deepcopy
from functools import partial
from pathlib import Path
import dill
import os
import toml
from importlib import import_module
from pydantic import ValidationError  # TODO - subject to decision on pydantic vs beartype for this
import warnings

# TODO - need to check and consolidate supported models
SupportedModels = (UAE, HiddenOutput, tf.keras.Sequential, tf.keras.Model)

logger = logging.getLogger(__name__)

Detector = Union[
    ChiSquareDrift,
    ClassifierDrift,
    ClassifierUncertaintyDrift,
    KSDrift,
    LearnedKernelDrift,
    LSDDDrift,
    MMDDrift,
    RegressorUncertaintyDrift,
    SpotTheDiffDrift,
    TabularDrift
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

LEARNED_DETECTOR = [
    ClassifierDrift.__name__,
    LearnedKernelDrift.__name__,
    SpotTheDiffDrift.__name__
]


# Fields to resolve in resolve_cfg ("resolve" meaning either load local artefact or resolve @registry, conversion to
# tuple, np.ndarray and np.dtype are dealt with separately).
FIELDS_TO_RESOLVE = [
    ['preprocess_fn'],
    ['preprocess_fn', 'function'],
    ['preprocess_fn', 'model'],
    ['preprocess_fn', 'embedding'],
    ['preprocess_fn', 'tokenizer'],
    ['preprocess_fn', 'device'],
    ['x_ref'],
    ['model'],
    ['kernel'],
    ['kernel', 'kernel']
]

# Directories to amend before resolving config (fields to prepend config file dir to)
DIR_FIELDS = [
    ['preprocess_fn'],
    ['preprocess_fn', 'function'],
    ['preprocess_fn', 'model', 'src'],
    ['preprocess_fn', 'embedding', 'src'],
    ['preprocess_fn', 'tokenizer', 'src'],
    ['x_ref'],
    ['model', 'src'],
    ['kernel'],
    ['kernel', 'kernel']
]

# Fields to convert from list to tuple in resolve_cfg
FIELDS_TO_TUPLE = [
    ['detector', 'kwargs', 'input_shape']
]

# Fields to convert from list to np.ndarray in resolve_cfg
FIELDS_TO_ARRAY = [
    ['sigma'],
    ['kernel', 'sigma']
]

# Fields to convert from str to np.dtype
FIELDS_TO_DTYPE = [
    ['preprocess_fn', 'dtype']
]


def validate_config(cfg: dict, resolved: bool = False) -> dict:
    # Get detector name
    if 'name' in cfg:
        detector_name = cfg['name']
    else:
        raise ValueError('`type` missing from config.toml.')

    # Validate detector specific config
    if detector_name in DETECTOR_CONFIGS.keys():
        if resolved:
            cfg = DETECTOR_CONFIGS_RESOLVED[detector_name](**cfg).dict()
        else:
            cfg = DETECTOR_CONFIGS[detector_name](**cfg).dict()
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


def load_detector_config(cfg: Union[str, os.PathLike, dict],
                         verbose: bool = False) -> Detector:
    # Load yaml if needed
    if isinstance(cfg, (str, os.PathLike)):
        config_file = Path(deepcopy(cfg))
        config_dir = config_file.parent
        cfg = read_detector_config(config_file)
    elif isinstance(cfg, dict):
        config_file = None
        config_dir = None
    else:
        raise ValueError('Detector `cfg` not recognised.')

    # Resolve and validate config
    cfg = validate_config(cfg)
    cfg = resolve_cfg(cfg, config_dir=config_dir, verbose=verbose)
    cfg = validate_config(cfg, resolved=True)

    # Backend and detector type
    detector_name = cfg.get('name')
    backend = cfg.pop('backend')  # popping so that cfg left as kwargs + `type` when passed to init_detector

    # Get x_ref
    x_ref = cfg.pop('x_ref')
    if isinstance(x_ref, str):  # If x_ref still a str, resolving must have failed
        raise ValueError("Failed to resolve x_ref field.")

    # Get kernel
    kernel = cfg.pop('kernel', None)  # Don't need to check if None as kernel=None defaults to GaussianRBF
    if isinstance(kernel, dict):
        kernel = load_kernel(kernel, cfg['device'])
    if detector_name == 'LearnedKernelDrift':
        if kernel is None:
            raise ValueError('A `kernel` must be specified for the LearnedKernelDrift detector.')
        elif not isinstance(kernel, DeepKernel):
            eps = cfg.pop('eps')  # TODO - default value?
            kernel = DeepKernel(kernel, eps=eps)

    # Get preprocess_fn
    preprocess_fn = cfg.pop('preprocess_fn')
    if isinstance(preprocess_fn, dict):
        preprocess_fn = load_preprocessor(preprocess_fn, backend=backend, verbose=verbose)

    # Get model
    model = cfg.pop('model', None)
    if model is not None:
        if not isinstance(model, SupportedModels):
            raise ValueError(f"Failed to load the {detector_name}'s model."
                             "Is the `model` field specified, and is the model a supported type?")

    # Init detector
    detector = init_detector(x_ref, cfg, preprocess_fn=preprocess_fn, model=model, kernel=kernel, backend=backend)

    # Update metadata
    detector.meta.update({'config_file': str(config_file.resolve())})

    return detector


def init_detector(x_ref: Union[np.ndarray, list],
                  cfg: dict,
                  preprocess_fn: Optional[Callable] = None,
                  model: Optional[SUPPORTED_MODELS] = None,
                  kernel: Optional[Callable] = None,
                  backend: Optional[str] = 'tensorflow') -> Detector:
    detector_name = cfg.pop('name', None)

    # Process args
    args = [x_ref]
    if model is not None:
        args.append(model)
    if detector_name in LEARNED_DETECTOR:
        raise NotImplementedError("Loading of learned detectors is not implemented yet.")  # TODO

    # Process kwargs (cfg should just be kwargs by this point)
    if detector_name in REQUIRES_BACKEND:
        cfg.update({'backend': backend})
    if preprocess_fn is not None:
        cfg.update({'preprocess_fn': preprocess_fn})
    if kernel is not None:
        cfg.update({'kernel': kernel})

    # Instantiate the detector
    detector = instantiate_class('alibi_detect.cd', detector_name, *args, **cfg)
    logger.info('Instantiated drift detector %s', detector_name)
    return detector


def load_kernel(cfg: dict, device: Optional[str] = None) -> Callable:
    """
    """
    kernel = cfg['kernel']
    sigma = cfg['sigma']
    if callable(kernel):
        if kernel == GaussianRBF_tf:
            sigma = tf.convert_to_tensor(sigma) if isinstance(sigma, np.ndarray) else sigma
            kernel = kernel(sigma=sigma, trainable=cfg['trainable'])
        elif kernel == GaussianRBF_torch:
            device = set_device(device)
            sigma = torch.from_numpy(sigma).to(device) if isinstance(sigma, np.ndarray) else None
            kernel = kernel(sigma=sigma, trainable=cfg['trainable'])
        else:
            kwargs = cfg['kwargs']
            kernel = kernel(**kwargs)
    else:
        raise ValueError('Unable to process kernel.)')
    return kernel


def load_preprocessor(cfg: dict,
                      backend: Optional[str] = 'tensorflow',
                      verbose: bool = False) -> Optional[Callable]:
    """
    This function builds a preprocess_fn from the preprocess dict in a detector config dict. The dict format is
    expected to match that generated by serialize_preprocess in alibi_detect.utils.saving. The model, tokenizer and
    preprocess_batch_fn are expected to be already resolved.
    """
    preprocess_fn = cfg.pop('function')

    if callable(preprocess_fn):
        if preprocess_fn == preprocess_drift_tf or preprocess_fn == preprocess_drift_torch:
            # If preprocess_drift function, kwargs is preprocess cfg minus 'function' and 'kwargs'
            cfg.pop('kwargs')
            kwargs = cfg.copy()

            # Handle embedding (if it exists)
            model = kwargs['model']
            emb = kwargs.pop('embedding')  # embedding passed to preprocess_drift as `model` therefore remove
            if emb is not None:
                if model is not None:
                    # If model exists, chain embedding and model together
                    if isinstance(model, UAE):
                        encoder = _Encoder(emb, mlp=model)
                        model = UAE(encoder_net=encoder)
                    else:
                        raise ValueError("Currently only model type 'UAE' is supported with an embedding.")
                else:
                    # if model doesn't exist, embedding is model
                    model = emb

            # Check model
            if model is None:
                raise ValueError("The 'model' field must be specified when 'preprocess_fn'='preprocess_drift'")
            kwargs.update({'model': model})

            # Backend specifics
            if backend == 'tensorflow':
                assert preprocess_fn == preprocess_drift_tf
                if not isinstance(model, KerasModel):
                    raise ValueError('The specified model is not a compatible tensorflow model.')
                kwargs.pop('device')
            elif backend == 'pytorch':
                assert preprocess_fn == preprocess_drift_torch
                if not isinstance(model, (nn.Module, nn.Sequential)):
                    raise ValueError('The specified model is not a compatible pytorch model.')
                device = cfg['device']
                if device is not None:
                    device = torch.device('cuda' if torch.cuda.is_available() else device)
                    kwargs.update({'model': kwargs['model'].to(device)})  # TODO - needs testing
                    kwargs.update({'device': device})
        else:
            kwargs = cfg['kwargs']  # If generic callable, kwargs is cfg['kwargs']

    else:
        logger.warning('Unable to process preprocess_fn. No preprocessing function is defined.')
        return None

    return partial(preprocess_fn, **kwargs)


def load_model(cfg: dict,
               backend: str,
               verbose: bool = False) -> SUPPORTED_MODELS:
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
        raise RuntimeError('Loading of pytorch models not currently supported')

    return model


def load_embedding(cfg: dict,
                   verbose: bool = False) -> TransformerEmbedding:
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


def load_tokenizer(cfg: dict,
                   verbose: bool = False) -> AutoTokenizer:
    src = cfg['src']
    kwargs = cfg['kwargs']
    if src is None:
        raise ValueError("No 'src' field for 'tokenizer'")
    src = Path(src)
    if not src.is_dir():
        raise FileNotFoundError("Compatible tokenizer not found at %s" % str(src.resolve()))

    tokenizer = AutoTokenizer.from_pretrained(src, **kwargs)
    return tokenizer


def get_nested_value(dic, keys):
    for key in keys:
        try:
            dic = dic[key]
        except (TypeError, KeyError):
            return None
    return dic


def set_nested_value(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def instantiate_class(module: str, name: str, *args, **kwargs) -> Detector:
    # TODO - need option to validate without instantiating class?
    klass = getattr(import_module(module), name)

    # Validate resolved detector args/kwargs
    val = None
    if hasattr(klass, 'validate'):
        val = klass.validate
    elif hasattr(klass.__init__, 'validate'):
        val = klass.__init__.validate
    if val is not None:
        try:
            val(*args, **kwargs)
        except ValidationError as exc:
            logger.warning("Error validating class instantiation arguments:\n%s" % exc)

    return klass(*args, **kwargs)


def read_detector_config(filepath: Union[os.PathLike, str]) -> dict:
    """
    This function reads a detector toml config file and returns a dict specifying the detector.
    """
    filepath = Path(filepath)
    cfg = toml.load(filepath)
    logger.info('Loaded config file from %s', str(filepath))
    return cfg


def resolve_cfg(cfg: dict, config_dir: Optional[Path], verbose: bool = False) -> dict:
    # Before main resolution, update filepaths relative to config file
    if config_dir is not None:
        for key in DIR_FIELDS:
            src = get_nested_value(cfg, key)
            if isinstance(src, str):
                src = config_dir.joinpath(Path(src))
                if src.is_file() or src.is_dir():
                    set_nested_value(cfg, key, str(src))

    # Resolve filepaths (load files) and resolve function/object registries
    for key in FIELDS_TO_RESOLVE:
        src = get_nested_value(cfg, key)

        obj = None
        # Resolve runtime registered function/object
        if isinstance(src, str):
            if src.startswith('@'):
                src = src[1:]
                if src in registry.get_all():
                    obj = registry.get(src)
                else:
                    raise ValueError("Can't find %s in the custom function registry" % src)
                if verbose:
                    logger.info('Successfully resolved registry entry %s' % src)

            # Load dill or numpy file
            elif Path(src).is_file():
                if Path(src).suffix == '.dill':
                    obj = dill.load(src)
                if Path(src).suffix == '.npy':
                    obj = np.load(src)

            # Pytorch device
            elif key[-1] == 'device':
                obj = set_device(src)

        # Resolve dict spec
        elif isinstance(src, dict):
            backend = cfg.get('backend', 'tensorflow')
            if key[-1] == 'model':
                obj = load_model(src, backend=backend, verbose=verbose)
            if key[-1] == 'embedding':
                obj = load_embedding(src, verbose=verbose)
            elif key[-1] == 'tokenizer':
                obj = load_tokenizer(src, verbose=verbose)

        # Put the resolved function into the cfg dict
        if obj is not None:
            set_nested_value(cfg, key, obj)

    # Convert selected lists to tuples
    for key in FIELDS_TO_TUPLE:
        val = get_nested_value(cfg, key)
        if val is not None:
            set_nested_value(cfg, key, tuple(val))

    # Convert selected lists to np.ndarray's
    for key in FIELDS_TO_ARRAY:
        val = get_nested_value(cfg, key)
        if val is not None:
            set_nested_value(cfg, key, np.array(val))

    # Convert selected str's to np.dtype's
    for key in FIELDS_TO_DTYPE:
        val = get_nested_value(cfg, key)
        if val is not None:
            val = val.split('.')
            val[0] = np if val[0] == 'np' else tf if val[0] == 'tf' else torch if val[0] == 'torch' else None
            if val[0] == None:
                raise ValueError("`dtype` must be in format np.<dtype>, tf.<dtype> or torch.<dtype>.")
            set_nested_value(cfg, key, getattr(val[0], val[1]))
    return cfg


def set_device(device: Optional[str] = None) -> torch.device:
    if device is None or device in ['gpu', 'cuda']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            logger.warning('No GPU detected, fall back on CPU.')
    else:
        device = torch.device('cpu')
    return device


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