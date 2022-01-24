# TODO - Need to modularise torch and tensorflow imports and use. e.g. has_tensorflow and has_pytorch etc
# TODO - clarify public vs private functions
# TODO - further modularisation? e.g. load_kernel_tf and load_kernel_torch? Or check that torch kernel isn't loaded
#  with torch installed etc elsewhere...
from alibi_detect.version import __version__
from alibi_detect.cd import ChiSquareDrift, KSDrift, MMDDrift, TabularDrift, LSDDDrift
from alibi_detect.cd import ClassifierUncertaintyDrift, RegressorUncertaintyDrift
from alibi_detect.cd import ClassifierDrift, LearnedKernelDrift, SpotTheDiffDrift
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_torch
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.utils.tensorflow.kernels import GaussianRBF as GaussianRBF_tf
from alibi_detect.utils.pytorch.kernels import GaussianRBF as GaussianRBF_torch
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.cd.tensorflow.preprocess import _Encoder
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.utils.registry import registry
from alibi_detect.utils.schemas import DETECTOR_CONFIGS, DETECTOR_CONFIGS_RESOLVED, SUPPORTED_MODELS, SupportedModels,\
    __config_spec__
from alibi_detect.utils.tensorflow.kernels import DeepKernel as DeepKernel_tf
from alibi_detect.utils.pytorch.kernels import DeepKernel as DeepKernel_torch
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
from pydantic import ValidationError
import warnings


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


def validate_config(cfg: dict, resolved: bool = False) -> dict:
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


def _load_detector_config(cfg: Union[str, os.PathLike, dict],
                          verbose: bool = False) -> Detector:
    # Load toml if needed
    if isinstance(cfg, (str, os.PathLike)):
        config_file = Path(deepcopy(cfg))
        config_dir = config_file.parent
        cfg = read_config(config_file)
        cfg = _replace(cfg, "None", None)  # TODO - move to read_config()
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
    backend = cfg.pop('backend')  # popping so that cfg left as kwargs + `name` when passed to init_detector

    # Get x_ref
    x_ref = cfg.pop('x_ref')
    if isinstance(x_ref, str):  # If x_ref still a str, resolving must have failed
        raise ValueError("Failed to resolve x_ref field.")

    # Get kernel
    kernel = cfg.pop('kernel', None)  # Don't need to check if None as kernel=None defaults to GaussianRBF
    if isinstance(kernel, dict):
        kernel = load_kernel(kernel, backend, cfg['device'])

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


def load_kernel(cfg: dict, backend: str = 'tensorflow', device: Optional[str] = None) -> Callable:
    """
    """

    if 'src' in cfg:  # Standard kernel config
        kernel = cfg['src']
        sigma = cfg['sigma']
        if callable(kernel):
            if kernel == GaussianRBF_tf:
                sigma = tf.convert_to_tensor(sigma) if isinstance(sigma, np.ndarray) else sigma
                kernel = kernel(sigma=sigma, trainable=cfg['trainable'])
            elif kernel == GaussianRBF_torch:
                torch_device = set_device(device)
                sigma = torch.from_numpy(sigma).to(torch_device) if isinstance(sigma, np.ndarray) else None
                kernel = kernel(sigma=sigma, trainable=cfg['trainable'])
            else:
                kwargs = cfg['kwargs']
                kernel = kernel(**kwargs)

    elif 'proj' in cfg:  # DeepKernel config
        proj = cfg['proj']
        eps = cfg['eps']
        # Kernel a
        kernel_a = cfg['kernel_a']
        if kernel_a is not None:
            kernel_a = load_kernel(kernel_a, backend, device)
        else:
            kernel_a = GaussianRBF_tf(trainable=True) if backend == 'tensorflow' else GaussianRBF_torch(trainable=True)
        # Kernel b
        kernel_b = cfg['kernel_b']
        if kernel_b is not None:
            kernel_b = load_kernel(kernel_b, backend, device)
        else:
            kernel_b = GaussianRBF_tf(trainable=True) if backend == 'tensorflow' else GaussianRBF_torch(trainable=True)

        # Assemble deep kernel
        if backend == 'tensorflow':
            kernel = DeepKernel_tf(proj, kernel_a=kernel_a, kernel_b=kernel_b, eps=eps)
        else:
            kernel = DeepKernel_torch(proj, kernel_a=kernel_a, kernel_b=kernel_b, eps=eps)
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
    preprocess_fn = cfg.pop('src')

    if callable(preprocess_fn):
        if preprocess_fn == preprocess_drift_tf or preprocess_fn == preprocess_drift_torch:
            # If preprocess_drift function, kwargs is preprocess cfg minus 'function' and 'kwargs'
            cfg.pop('kwargs')
            kwargs = cfg.copy()

            # Final processing of model (and/or embedding)
            model = kwargs['model']
            emb = kwargs.pop('embedding')  # embedding passed to preprocess_drift as `model` therefore remove
            model = prep_model_and_embedding(model, emb, backend=backend)
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
        raise NotImplementedError('Loading of pytorch models not currently supported')

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


def load_optimizer(cfg: dict,
                   backend: str,
                   verbose: bool = False) -> Union[tf.keras.optimizers.Optimizer, Callable]:

    if backend == 'tensorflow':
        optimizer = tf.keras.optimizers.deserialize(cfg)
    else:
        raise NotImplementedError('Loading of pytorch optimizers not currently supported')

    return optimizer


def prep_model_and_embedding(model: Optional[SUPPORTED_MODELS], emb: Optional[TransformerEmbedding],
                             backend: str) -> SUPPORTED_MODELS:
    """
    Function to perform final preprocessing of model before it is passed to preprocess_drift. This is separated from
    load_model in order to reduce complexity of load functions (with future model load functionality in mind), and also
    to separate embedding logic from model loading (allows for cleaner config layout and resolution of it).

    Note: downside of separating this function from load_model is we no longer know model_type = UAE/HiddenState/custom
     when chaining embedding and model together. Currently requires a bit of a hack to avoid nesting UAE's.
     On plus side, makes clearer distinction between settings for model loading and settings for combining
     model/embedding (the latter going in preprocess_fn config). But need to revisit...

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
            model = model.encoder if isinstance(model, UAE) else model
            if emb is not None:
                model = _Encoder(emb, mlp=model)
            model = UAE(encoder_net=model)

        else:
            raise NotImplementedError("Loading of pytorch models is not currently implemented.")

    # If no model exists, store embedding in model (both may be None)
    else:
        model = emb

    return model


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


def read_config(filepath: Union[os.PathLike, str]) -> dict:
    """
    This function reads a detector toml config file and returns a dict specifying the detector.
    """
    filepath = Path(filepath)
    cfg = toml.load(filepath)
    logger.info('Loaded config file from %s', str(filepath))

    # Convert keys in categories_per_feature back to str
    # TODO - this is undesirable, but currently necessary in order to use toml library, as error with toml dumping dict
    #  with int keys.
    if 'categories_per_feature' in cfg:
        new = {}
        for key in cfg['categories_per_feature']:
            new[int(key)] = cfg['categories_per_feature'][key]
        cfg['categories_per_feature'] = new

    return cfg  # type: ignore[return-value] # TODO - toml actually returns MutableMapping, consider updating throughout


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
                    obj = dill.load(open(src, 'rb'))
                if Path(src).suffix == '.npy':
                    obj = np.load(src)

            # Pytorch device
            elif key[-1] == 'device':
                obj = set_device(src)

        # Resolve dict spec
        elif isinstance(src, dict):
            backend = cfg.get('backend', 'tensorflow')
            if key[-1] in ('model', 'proj'):
                obj = load_model(src, backend=backend, verbose=verbose)
            if key[-1] == 'embedding':
                obj = load_embedding(src, verbose=verbose)
            elif key[-1] == 'tokenizer':
                obj = load_tokenizer(src, verbose=verbose)
            elif key[-1] == 'optimizer':
                obj = load_optimizer(src, backend=backend, verbose=verbose)

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
            if val[0] is None:
                raise ValueError("`dtype` must be in format np.<dtype>, tf.<dtype> or torch.<dtype>.")
            set_nested_value(cfg, key, getattr(val[0], val[1]))

    return cfg


def set_device(device: Optional[str]) -> torch.device:
    if device is None or device in ['gpu', 'cuda']:
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch_device.type == 'cpu':
            logger.warning('No GPU detected, fall back on CPU.')
    else:
        torch_device = torch.device('cpu')
    return torch_device


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
    for k, v in cfg.items():
        if v == orig:
            cfg[k] = new
        elif isinstance(v, dict):
            _replace(v, orig, new)
    return cfg
