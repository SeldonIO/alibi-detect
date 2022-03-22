# TODO - Need to modularise torch and tensorflow imports and use. e.g. has_tensorflow and has_pytorch etc
from alibi_detect.utils.registry import registry
from alibi_detect.utils.schemas import SupportedModels  # type: ignore[attr-defined]
from alibi_detect.utils.validate import validate_config
from alibi_detect.utils.tensorflow._loading import load_model as load_model_tf, \
    prep_model_and_emb as prep_model_and_emb_tf, load_kernel_config as load_kernel_config_tf, \
    load_optimizer as load_optimizer_tf, load_embedding as load_embedding_tf, load_detector_legacy, Detectors
#  from alibi_detect.utils.pytorch._loading import set_device
import tensorflow as tf  # TODO - this is currently only required for FIELDS_TO_DTYPE conversion. Remove in future.
from transformers import AutoTokenizer
import numpy as np
from typing import Union, Optional, Callable, Any
from copy import deepcopy
from functools import partial
from pathlib import Path
import dill
import os
import toml
from importlib import import_module
import logging

logger = logging.getLogger(__name__)

REQUIRES_BACKEND = [
    'ClassifierDrift',
    'ClassifierUncertaintyDrift',
    'LearnedKernelDrift',
    'LSDDDrift',
    'MMDDrift',
    'RegressorUncertaintyDrift',
    'SpotTheDiffDrift'
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


def load_detector(filepath: Union[str, os.PathLike], **kwargs) -> Detectors:
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
            return load_detector_legacy(filepath, '.dill', **kwargs)
        elif 'meta.pickle' in files:
            return load_detector_legacy(filepath, '.pickle', **kwargs)
        else:
            raise ValueError('Neither meta.dill, meta.pickle or config.toml exist in {}.'.format(filepath))

    # No other file types are accepted, so if not dir raise error
    else:
        raise ValueError("load_detector accepts only a filepath to a directory, or a config.toml file.")


# TODO - will eventually become load_detector
def _load_detector_config(cfg: Union[str, os.PathLike, dict]) -> Detectors:
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
        kernel = _load_kernel_config(kernel, backend, cfg['device'])

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


def _init_detector(x_ref: Union[np.ndarray, list],
                   cfg: dict,
                   preprocess_fn: Optional[Callable] = None,
                   model: Optional[Callable] = None,
                   kernel: Optional[Callable] = None,
                   backend: Optional[str] = 'tensorflow') -> Detectors:
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
    version_warning = cfg.pop('version_warning', False)

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
    detector.meta['version_warning'] = version_warning  # Insert here to avoid needing to add as kwarg
    logger.info('Instantiated drift detector %s', detector_name)
    return detector


def _load_kernel_config(cfg: dict, backend: str = 'tensorflow', device: Optional[str] = None) -> Callable:
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
    if backend == 'tensorflow':
        kernel = load_kernel_config_tf(cfg)
    else:
        kernel = None
        # kernel = load_kernel_config_torch(cfg, device)
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
        if preprocess_fn.__name__ == 'preprocess_drift':
            # If preprocess_drift function, kwargs is preprocess cfg minus 'function' and 'kwargs'
            cfg.pop('kwargs')
            kwargs = cfg.copy()

            # Final processing of model (and/or embedding)
            model = kwargs['model']
            emb = kwargs.pop('embedding')  # embedding passed to preprocess_drift as `model` therefore remove

            # Backend specifics
            if backend == 'tensorflow':
                model = prep_model_and_emb_tf(model, emb)
                kwargs.pop('device')
            elif backend == 'pytorch':  # TODO - once optional deps implemented
                raise NotImplementedError('Loading preprocess_fn for PyTorch not yet supported.')
#               device = cfg['device'] # TODO - device should be set already - check
#               kwargs.update({'model': kwargs['model'].to(device)})  # TODO - need .to(device) here?
#               kwargs.update({'device': device})
            kwargs.update({'model': model})
        else:
            kwargs = cfg['kwargs']  # If generic callable, kwargs is cfg['kwargs']

    else:
        logger.warning('Unable to process preprocess_fn. No preprocessing function is defined.')
        return None

    return partial(preprocess_fn, **kwargs)


def _load_model_config(cfg: dict,
                       backend: str) -> Callable:
    """
    Loads TensorFlow, PyTorch and scikit-learn models (currently only TensorFlow supported), from a model config
    dict.

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
    src = Path(src)
    if not src.is_dir():
        raise FileNotFoundError("Compatible model not found at %s" % str(src.resolve()))

    if backend == 'tensorflow':
        model = load_model_tf(src, load_dir='.', custom_objects=custom_obj, typ=typ)
    else:
        raise NotImplementedError('Loading of pytorch models not currently supported')

    return model


def _load_embedding(cfg: dict, backend: str) -> Callable:  # TODO: Could type return more tightly
    """
    Load a pre-trained text embedding from an embedding config dict.

    Parameters
    ----------
    cfg
        An embedding config dict. (see the pydantic schemas).
    backend
        The backend.

    Returns
    -------
    The loaded embedding.
    """
    src = cfg['src']
    layers = cfg['layers']
    typ = cfg['type']
    if backend == 'tensorflow':
        emb = load_embedding_tf(src, embedding_type=typ, layers=layers)
    else:
        emb = None
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
    src = Path(src)
    tokenizer = AutoTokenizer.from_pretrained(src, **kwargs)
    return tokenizer


def _load_optimizer(cfg: dict,
                    backend: str) -> Union['tf.keras.optimizers.Optimizer', Callable]:
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
        optimizer = load_optimizer_tf(cfg)
    else:
        raise NotImplementedError('Loading of pytorch optimizers not currently supported')
    return optimizer


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

    cfg = dict(toml.load(filepath))  # toml.load types return as MutableMapping, force to dict
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

    return cfg


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
#                obj = set_device(src)

        # Resolve dict spec
        elif isinstance(src, dict):
            backend = cfg.get('backend', 'tensorflow')
            if key[-1] in ('model', 'proj'):
                obj = _load_model_config(src, backend)
            if key[-1] == 'embedding':
                obj = _load_embedding(src, backend)
            elif key[-1] == 'tokenizer':
                obj = _load_tokenizer(src)
            elif key[-1] == 'optimizer':
                obj = _load_optimizer(src, backend)

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

    # Convert selected str's to required dtype's
    for key in FIELDS_TO_DTYPE:
        val = _get_nested_value(cfg, key)
        if val is not None:
            val = val.split('.')
#            val[0] = np if val[0] == 'np' else tf if val[0] == 'tf' else torch if val[0] == 'torch' else None
            # TODO - add above back in once optional deps are handled properly
            val[0] = np if val[0] == 'np' else tf if val[0] == 'tf' else None
            if val[0] is None:
                raise ValueError("`dtype` must be in format np.<dtype>, tf.<dtype> or torch.<dtype>.")
            _set_nested_value(cfg, key, getattr(val[0], val[1]))

    return cfg


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
