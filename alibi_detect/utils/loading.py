from alibi_detect.version import __version__
from alibi_detect.cd import ChiSquareDrift, KSDrift, MMDDrift, TabularDrift, LSDDDrift
from alibi_detect.cd import ClassifierUncertaintyDrift, RegressorUncertaintyDrift
from alibi_detect.cd import ClassifierDrift, LearnedKernelDrift, SpotTheDiffDrift
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_torch
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.cd.tensorflow import HiddenOutput, UAE
from alibi_detect.cd.tensorflow.preprocess import _Encoder
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.utils.custom import custom_artefact
from alibi_detect.utils.tensorflow.kernels import DeepKernel
import numpy as np
from transformers import AutoTokenizer
from torch import device as torch_device
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
from ruamel.yaml import YAML
from importlib import import_module
from pydantic import ValidationError  # TODO - subject to decision on pydantic
import warnings

# TODO - need to check and consolidate supported models
SUPPORTED_MODELS = Union[UAE, HiddenOutput, tf.keras.Sequential, tf.keras.Model]
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

REQUIRES_MODEL = [
    ClassifierUncertaintyDrift.__name__,
    RegressorUncertaintyDrift.__name__
]

OPTIONAL_PREPROCESS = [
    ChiSquareDrift.__name__,
    KSDrift.__name__,
    LSDDDrift.__name__,
    MMDDrift.__name__,
    TabularDrift.__name__
]

LEARNED_DETECTOR = [
    ClassifierDrift.__name__,
    LearnedKernelDrift.__name__,
    SpotTheDiffDrift.__name__
]

FIELDS_TO_RESOLVE = [
    ['x_ref'],
    ['model'],
    ['preprocess', 'preprocess_fn'],
    ['preprocess', 'kwargs', 'model'],
    ['preprocess', 'kwargs', 'embedding'],
    ['preprocess', 'kwargs', 'tokenizer'],
    ['detector', 'kwargs', 'kernel']  # TODO - where should kernel be spec'd?
]


def load_detector_config(cfg: Union[str, os.PathLike, dict],
                         verbose: bool = False) -> Detector:
    # Load yaml if needed
    if isinstance(cfg, (str, os.PathLike)):
        config_file = Path(deepcopy(cfg)).resolve()
        cfg = read_detector_config(config_file)
    elif isinstance(cfg, dict):
        config_file = None
    else:
        raise ValueError('Detector `cfg` not recognised.')

    # check version
    version = cfg.get('version')
    if version is not None and version != __version__:
        warnings.warn(f'Trying to load detector from version {version} when using version '
                      f'{__version__}. This may lead to breaking code or invalid results.')

    # Resolve cfg
    cfg.update({'registries': {}})
    cfg = resolve_cfg(cfg, verbose=verbose)

    # Load detector config
    if 'detector' in cfg:
        detector_cfg = cfg['detector']
        detector_name = cfg.get('type', None)
    else:
        raise ValueError("Config file must contain a 'detector' field.")
    if detector_name is None:
        raise ValueError("Config file must contain a 'detector'->'type' field.")
    backend = cfg.setdefault('backend', 'tensorflow')

    # x_ref
    x_ref = cfg['x_ref']
    if isinstance(x_ref, str):  # If x_ref still a str, resolving must have failed
        raise ValueError("Failed to resolve x_ref field.")

    # Load preprocessor if specified
    if detector_name in OPTIONAL_PREPROCESS and 'preprocess' in cfg:
        preprocessor_cfg = cfg['preprocess']
        preprocess_fn = load_preprocessor(preprocessor_cfg, backend=backend, verbose=verbose)
    else:
        preprocess_fn = None

    # Load model (model should have already been resolved by this point, so just check its not a str)
    model = None
    if detector_name in REQUIRES_MODEL:
        model = detector_cfg.get('model', None)
        if not isinstance(model, SupportedModels):
            raise ValueError(f"Failed to load the {detector_name}'s model."
                             "Is the `model` field specified, and is the model a supported type?")
#TODO        if detector_name == 'LearnedKernelDrift':
#TODO            eps =
#TODO            model = DeepKernel(model, eps=0.01)

    # Init detector
    detector = init_detector(x_ref, detector_cfg, preprocess_fn=preprocess_fn, model=model, backend=backend)

    # Update metadata
#    detector.meta.update({'config': cfg, 'config_file': config_file})
    detector.meta.update({'config_file': config_file, 'registries': cfg['registries']})

    return detector


# TODO - need to set device in here too (for pytorch)
def init_detector(x_ref: Union[np.ndarray, list],
                  cfg: dict,
                  preprocess_fn: Optional[Callable] = None,
                  model: Optional[SUPPORTED_MODELS] = None,
                  backend: Optional[str] = 'tensorflow') -> Detector:

    detector_name = cfg.get('type', None)
    if detector_name is None:
        raise ValueError('The detector `type` must be specified.')

    # Process args
    args = [x_ref]
    if model is not None:
        args.append(model)
    if detector_name in LEARNED_DETECTOR:
        raise NotImplementedError("Loading of learned detectors is not implemented yet.")  # TODO

    # Process kwargs
    kwargs = cfg.get('kwargs', {})
    if detector_name in REQUIRES_BACKEND:
        kwargs.update({'backend': backend})
    if detector_name in OPTIONAL_PREPROCESS:
        if preprocess_fn is not None:
            kwargs.update({'preprocess_fn': preprocess_fn})

    # Instantiate the detector
    detector = instantiate_class('alibi_detect.cd', detector_name, *args, **kwargs)
    logger.info('Instantiated drift detector %s', detector_name)
    return detector


def load_preprocessor(cfg: dict,
                      backend: Optional[str] = 'tensorflow',
                      verbose: bool = False) -> Optional[Callable]:
    """
    This function builds a preprocess_fn from the preprocess dict in a detector config dict. The dict format is
    expected to match that generated by serialize_preprocess in alibi_detect.utils.saving. The model, tokenizer and
    preprocess_batch_fn are expected to be already resolved.
    """
    kwargs = cfg.get('kwargs', {})
    preprocess_fn = cfg.get('preprocess_fn')

    # If string...
    if isinstance(preprocess_fn, str):
        # If still a str, check if this refers to in-built preprocess_drift function
        if preprocess_fn == 'preprocess_drift':
            if 'model' not in kwargs:
                raise ValueError("The 'model' field must be specified when 'preprocess_fn'='preprocess_drift'")
            if backend == 'tensorflow':
                preprocess_fn = preprocess_drift_tf
                if not isinstance(kwargs['model'], KerasModel):
                    raise ValueError('The specified model is not a compatible tensorflow model.')
            elif backend == 'pytorch':
                preprocess_fn = preprocess_drift_torch
                if not isinstance(kwargs['model'], (nn.Module, nn.Sequential)):
                    raise ValueError('The specified model is not a compatible pytorch model.')
                if 'device' in cfg:
                    device = torch_device(kwargs['device'])
                    kwargs.update({'model': kwargs['model'].to(device)})  # TODO - needs testing
                    kwargs.update({'device': device})

        else:
            # If still a str, but not 'preprocess_drift', resolution of local filepath must have failed
            raise ValueError("If preprocess_fn is a str, it must either be a filepath to a .dill file, "
                             "or 'preprocess_drift'")

    # If already function, assume preprocess_fn is an already resolved function (TODO: tighten this check?)
    elif callable(preprocess_fn):
        pass
    else:
        logger.warning('Unable to process preprocess_fn. No preprocessing function is defined.')
        return None

    return partial(preprocess_fn, **kwargs)


def load_model(cfg: dict,
               backend: str,
               verbose: bool = False) \
        -> Union[tf.keras.Model, nn.Module, nn.Sequential, TransformerEmbedding]:
    # Load model
    src = cfg.get('src', None)
    typ = cfg.get('type', 'custom')
    custom_obj = cfg.get('custom_objects', None)
    if src is None:
        model = None
        if verbose:
            logging.warning("No 'src' field for 'model'. A model will not be loaded.")
    else:
        src = Path(src)
        if backend == 'tensorflow':
            model = load_tf_model(src, load_dir='.', custom_objects=custom_obj)
            if typ == 'UAE':
                model = UAE(encoder_net=model)
            elif typ == 'HiddenOutput' or typ == 'custom':
                pass
            else:
                raise ValueError("Model 'type' not recognised.")
        else:
            raise ValueError('Loading of pytorch models not currently supported')

    # Load embedding
    cfg_embed = cfg.get('embedding', None)
    if cfg_embed is not None:
        cfg_embed = cfg_embed.copy()
        filepath = cfg_embed.pop('src')
        emb = TransformerEmbedding(filepath, **cfg_embed)

        if model is not None:
            # If model exists, chain embedding and model together
            if typ == 'UAE':
                encoder = _Encoder(emb, mlp=model)
                model = UAE(encoder_net=encoder)
            else:
                raise ValueError("Currently only model type 'UAE' is supported with an embedding.")
        else:
            # if model doesn't exist, embedding is model
            model = emb

    return model


def load_tokenizer(cfg: dict,
                   verbose: bool = False) -> Callable:
    cfg = cfg.copy()
    src = cfg.pop('src')
    tokenizer = AutoTokenizer.from_pretrained(src, **cfg)
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

    # Validate
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
    This function reads a detector yaml config file and returns a dict specifying the detector.
    """
    yaml = YAML(typ='safe')
    filepath = Path(filepath)
    cfg = yaml.load(filepath)
    logger.info('Loaded config file from %s', str(filepath))
    return cfg


def resolve_cfg(cfg: dict, verbose: bool = False) -> dict:
    for key in FIELDS_TO_RESOLVE:
        src = get_nested_value(cfg, key)

        obj = None
        # Resolve runtime registered function/object
        if isinstance(src, str):
            if src.startswith('@'):
                src = src[1:]
                if src in custom_artefact.get_all():
                    obj = custom_artefact.get(src)
                    cfg['registries'].update({src: custom_artefact.find(src)})
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

        # Resolve dict spec
        elif isinstance(src, dict):
            backend = cfg.get('backend', 'tensorflow')
            if key[-1] == 'model':
                obj = load_model(src, backend=backend, verbose=verbose)
            elif key[-1] == 'tokenizer':
                obj = load_tokenizer(src, verbose=verbose)
            if obj is None:
                raise ValueError('Failed to process %s dict' % key[-1])

        # Put the resolved function into the cfg dict
        if obj is not None:
            set_nested_value(cfg, key, obj)

    # Hard code tuples (yaml reads in as lists)
    keys = [['detector', 'kwargs', 'input_shape']]
    for key in keys:
        val = get_nested_value(cfg, key)
        set_nested_value(cfg, key, tuple(val))

    return cfg


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
