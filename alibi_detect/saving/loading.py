import logging
import os
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional, Union, Type, TYPE_CHECKING

import dill
import numpy as np
import toml
from transformers import AutoTokenizer

from alibi_detect.saving.registry import registry
from alibi_detect.saving._tensorflow import load_detector_legacy, load_embedding_tf, load_kernel_config_tf, \
    load_model_tf, load_optimizer_tf, prep_model_and_emb_tf, get_tf_dtype
from alibi_detect.saving._pytorch import load_embedding_pt, load_kernel_config_pt, load_model_pt, \
    load_optimizer_pt, prep_model_and_emb_pt, get_pt_dtype
from alibi_detect.saving._keops import load_kernel_config_ke
from alibi_detect.saving._sklearn import load_model_sk
from alibi_detect.saving.validate import validate_config
from alibi_detect.base import Detector, ConfigurableDetector, StatefulDetectorOnline
from alibi_detect.utils.frameworks import has_tensorflow, has_pytorch, Framework
from alibi_detect.saving.schemas import supported_models_tf, supported_models_torch
from alibi_detect.utils.missing_optional_dependency import import_optional
get_device = import_optional('alibi_detect.utils.pytorch.misc', names=['get_device'])

if TYPE_CHECKING:
    import tensorflow as tf
    import torch

STATE_PATH = 'state/'  # directory (relative to detector directory) where state is saved (and loaded from)

logger = logging.getLogger(__name__)


# Fields to resolve in resolve_config ("resolve" meaning either load local artefact or resolve @registry, conversion to
# tuple, np.ndarray and np.dtype are dealt with separately).
# Note: For fields consisting of nested dicts, they must be listed in order from deepest to shallowest, so that the
# deepest fields are resolved first. e.g. 'preprocess_fn.src' must be resolved before 'preprocess_fn'.
FIELDS_TO_RESOLVE = [
    ['preprocess_fn', 'src'],
    ['preprocess_fn', 'model'],
    ['preprocess_fn', 'embedding'],
    ['preprocess_fn', 'tokenizer'],
    ['preprocess_fn', 'preprocess_batch_fn'],
    ['preprocess_fn'],
    ['x_ref'],
    ['c_ref'],
    ['model'],
    ['optimizer'],
    ['reg_loss_fn'],
    ['dataset'],
    ['kernel', 'src'],
    ['kernel', 'proj'],
    ['kernel', 'init_sigma_fn'],
    ['kernel', 'kernel_a', 'src'],
    ['kernel', 'kernel_a', 'init_sigma_fn'],
    ['kernel', 'kernel_b', 'src'],
    ['kernel', 'kernel_b', 'init_sigma_fn'],
    ['kernel'],
    ['x_kernel', 'src'],
    ['x_kernel', 'init_sigma_fn'],
    ['x_kernel'],
    ['c_kernel', 'src'],
    ['c_kernel', 'init_sigma_fn'],
    ['c_kernel'],
    ['initial_diffs'],
    ['tokenizer']
]

# Fields to convert from str to dtype
FIELDS_TO_DTYPE = [
    ['preprocess_fn', 'dtype']
]


def load_detector(filepath: Union[str, os.PathLike], enable_unsafe_loading: bool = False,
                  **kwargs) -> Union[Detector, ConfigurableDetector]:
    """
    Load outlier, drift or adversarial detector.

    Parameters
    ----------
    filepath
        Load directory.
    enable_unsafe_loading
        Sets allow_pickle=True when a np.ndarray is loaded from a .npy file referenced in the detector config. Needed
        if you have to load objects.
        Only applied if the filepath is config.toml or a directory containing a config.toml.
        It has security implications: https://nvd.nist.gov/vuln/detail/cve-2019-6446.

    Returns
    -------
    Loaded outlier or adversarial detector object.
    """
    filepath = Path(filepath)
    # If reference is a 'config.toml' itself, pass to new load function
    if filepath.name == 'config.toml':
        return _load_detector_config(filepath, enable_unsafe_loading=enable_unsafe_loading)

    # Otherwise, if a directory, look for meta.dill, meta.pickle or config.toml inside it
    elif filepath.is_dir():
        files = [str(f.name) for f in filepath.iterdir() if f.is_file()]
        if 'config.toml' in files:
            return _load_detector_config(filepath.joinpath('config.toml'), enable_unsafe_loading=enable_unsafe_loading)
        elif 'meta.dill' in files:
            return load_detector_legacy(filepath, '.dill', **kwargs)
        elif 'meta.pickle' in files:
            return load_detector_legacy(filepath, '.pickle', **kwargs)
        else:
            raise ValueError(f'Neither meta.dill, meta.pickle or config.toml exist in {filepath}.')

    # No other file types are accepted, so if not dir raise error
    else:
        raise ValueError("load_detector accepts only a filepath to a directory, or a config.toml file.")


# TODO - will eventually become load_detector
def _load_detector_config(filepath: Union[str, os.PathLike], enable_unsafe_loading: bool = False) \
        -> ConfigurableDetector:
    """
    Loads a drift detector specified in a detector config dict. Validation is performed with pydantic.

    Parameters
    ----------
    filepath
        Filepath to the `config.toml` file.
    enable_unsafe_loading
        Sets allow_pickle=True when a np.ndarray is loaded from a .npy file (happens if the .toml references one).
        Needed if you have to load objects.
        It has security implications: https://nvd.nist.gov/vuln/detail/cve-2019-6446

    Returns
    -------
    The instantiated detector.
    """
    # Load toml if needed
    if isinstance(filepath, (str, os.PathLike)):
        config_file = Path(filepath)
        config_dir = config_file.parent
        cfg = read_config(config_file)
    else:
        raise ValueError("`filepath` should point to a directory containing a 'config.toml' file.")

    # Resolve and validate config
    cfg = validate_config(cfg)
    logger.info('Validated unresolved config.')
    cfg = resolve_config(cfg, config_dir=config_dir, enable_unsafe_loading=enable_unsafe_loading)
    cfg = validate_config(cfg, resolved=True)
    logger.info('Validated resolved config.')

    # Init detector from config
    logger.info('Instantiating detector.')
    detector = _init_detector(cfg)

    # Load state if it exists (and detector supports it)
    # TODO - this will be removed in follow-up offline state PR, as loading to be moved to __init__ (w/ state_dir kwarg)
    if isinstance(detector, StatefulDetectorOnline):
        state_dir = config_dir.joinpath(STATE_PATH)
        if state_dir.is_dir():
            detector.load_state(state_dir)

    logger.info('Finished loading detector.')

    return detector


def _init_detector(cfg: dict) -> ConfigurableDetector:
    """
    Instantiates a detector from a fully resolved config dictionary.

    Parameters
    ----------
    cfg
        The detector's resolved config dictionary.

    Returns
    -------
    The instantiated detector.
    """
    detector_name = cfg.pop('name')

    # Instantiate the detector
    klass = getattr(import_module('alibi_detect.cd'), detector_name)
    detector = klass.from_config(cfg)
    logger.info('Instantiated drift detector {}'.format(detector_name))
    return detector


def _load_kernel_config(cfg: dict, backend: str = Framework.TENSORFLOW) -> Callable:
    """
    Loads a kernel from a kernel config dict.

    Parameters
    ----------
    cfg
        A kernel config dict. (see pydantic schema's).
    backend
        The backend.

    Returns
    -------
    The kernel.
    """
    if backend == Framework.TENSORFLOW:
        kernel = load_kernel_config_tf(cfg)
    elif backend == Framework.PYTORCH:
        kernel = load_kernel_config_pt(cfg)
    else:  # backend=='keops'
        kernel = load_kernel_config_ke(cfg)
    return kernel


def _load_preprocess_config(cfg: dict) -> Optional[Callable]:
    """
    This function builds a preprocess_fn from the preprocess dict in a detector config dict. The dict format is
    expected to match that generated by serialize_preprocess in alibi_detect.utils.saving (also see pydantic schema).
    The model, tokenizer and preprocess_batch_fn are expected to be already resolved.

    Parameters
    ----------
    cfg
        A preprocess_fn config dict. (see pydantic schemas).

    Returns
    -------
    The preprocess_fn function.
    """
    preprocess_fn = cfg.pop('src')

    if callable(preprocess_fn):
        if preprocess_fn.__name__ == 'preprocess_drift':
            # If preprocess_drift function, kwargs is preprocess cfg minus 'src' and 'kwargs'
            cfg.pop('kwargs')
            kwargs = cfg.copy()

            # Final processing of model (and/or embedding)
            model = kwargs['model']
            emb = kwargs.pop('embedding')  # embedding passed to preprocess_drift as `model` therefore remove

            # Backend specifics
            if has_tensorflow and isinstance(model, supported_models_tf):
                model = prep_model_and_emb_tf(model, emb)
            elif has_pytorch and isinstance(model, supported_models_torch):
                model = prep_model_and_emb_pt(model, emb)
            elif model is None:
                model = emb
            if model is None:
                raise ValueError("A 'model'  and/or `embedding` must be specified when "
                                 "preprocess_fn='preprocess_drift'")
            kwargs.update({'model': model})
            # Set`device` if a PyTorch model, otherwise remove from kwargs
            if isinstance(model, supported_models_torch):
                device = get_device(cfg['device'])
                model = model.to(device).eval()
                kwargs.update({'device': device})
                kwargs.update({'model': model})
            else:
                kwargs.pop('device')
        else:
            kwargs = cfg['kwargs']  # If generic callable, kwargs is cfg['kwargs']

    else:
        logger.warning('Unable to process preprocess_fn. No preprocessing function is defined.')
        return None

    if kwargs == {}:
        return preprocess_fn
    else:
        return partial(preprocess_fn, **kwargs)


def _load_model_config(cfg: dict) -> Callable:
    """
    Loads supported models from a model config dict.

    Parameters
    ----------
    cfg
        Model config dict. (see pydantic model schemas).

    Returns
    -------
    The loaded model.
    """

    # Load model
    flavour = cfg['flavour']
    src = cfg['src']
    custom_obj = cfg['custom_objects']
    layer = cfg['layer']
    src = Path(src)
    if not src.is_dir():
        raise FileNotFoundError("The `src` field is not a recognised directory. It should be a directory containing "
                                "a compatible model.")

    if flavour == Framework.TENSORFLOW:
        model = load_model_tf(src, custom_objects=custom_obj, layer=layer)
    elif flavour == Framework.PYTORCH:
        model = load_model_pt(src, layer=layer)
    elif flavour == Framework.SKLEARN:
        model = load_model_sk(src)

    return model


def _load_embedding_config(cfg: dict) -> Callable:  # TODO: Could type return more tightly
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
    flavour = cfg['flavour']
    if flavour == Framework.TENSORFLOW:
        emb = load_embedding_tf(src, embedding_type=typ, layers=layers)
    else:
        emb = load_embedding_pt(src, embedding_type=typ, layers=layers)
    return emb


def _load_tokenizer_config(cfg: dict) -> AutoTokenizer:
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


def _load_optimizer_config(cfg: dict, backend: str) \
        -> Union['tf.keras.optimizers.Optimizer', Type['tf.keras.optimizers.Optimizer'],
                 Type['torch.optim.Optimizer']]:
    """
    Loads an optimzier from an optimizer config dict.

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
    if backend == Framework.TENSORFLOW:
        return load_optimizer_tf(cfg)
    else:
        return load_optimizer_pt(cfg)


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


def _set_dtypes(cfg: dict):
    """
    Converts str's in the config dictionary to dtypes e.g. 'np.float32' is converted to np.float32.

    Parameters
    ----------
    cfg
        The config dictionary.
    """
    # TODO - we could explore a custom pydantic generic type for this (similar to how we handle NDArray)
    for key in FIELDS_TO_DTYPE:
        val = _get_nested_value(cfg, key)
        if val is not None:
            lib, dtype, *_ = val.split('.')
            # val[0] = np if val[0] == 'np' else tf if val[0] == 'tf' else torch if val[0] == 'torch' else None
            # TODO - add above back in once optional deps are handled properly
            if lib is None:
                raise ValueError("`dtype` must be in format np.<dtype>, tf.<dtype> or torch.<dtype>.")
            {
                'tf': lambda: _set_nested_value(cfg, key, get_tf_dtype(dtype)),
                'torch': lambda: _set_nested_value(cfg, key, get_pt_dtype(dtype)),
                'np': lambda: _set_nested_value(cfg, key, getattr(np, dtype)),
            }[lib]()


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
    logger.info('Loaded config file from {}'.format(str(filepath)))

    # This is necessary as no None/null in toml spec., and missing values are set to defaults set in pydantic models.
    # But we sometimes need to explicitly spec as None.
    cfg = _replace(cfg, "None", None)

    return cfg


def resolve_config(cfg: dict, config_dir: Optional[Path], enable_unsafe_loading: bool = False) -> dict:
    """
    Resolves artefacts in a config dict. For example x_ref='x_ref.npy' is resolved by loading the np.ndarray from
    the .npy file. For a list of fields that are resolved, see
    https://docs.seldon.io/projects/alibi-detect/en/stable/overview/config_file.html.

    Parameters
    ----------
    cfg
        The unresolved config dict.
    config_dir
        Filepath to directory the `config.toml` is located in. Only required if different from the
        runtime directory, and artefacts are specified with filepaths relative to the config.toml file.
    enable_unsafe_loading
        If set to true, allow_pickle=True is set in np.load(). Needed if you have to load objects.
        It has security implications: https://nvd.nist.gov/vuln/detail/cve-2019-6446

    Returns
    -------
    The resolved config dict.
    """
    # Convert selected str's to required dtype's (all other type coercion is performed by pydantic)
    _set_dtypes(cfg)

    # Before main resolution, update filepaths relative to config file
    if config_dir is not None:
        _prepend_cfg_filepaths(cfg, config_dir)

    # Resolve filepaths (load files) and resolve function/object registries
    for key in FIELDS_TO_RESOLVE:
        logger.info('Resolving config field: {}.'.format(key))
        src = _get_nested_value(cfg, key)
        obj = None

        # Resolve string references to registered objects and filepaths
        if isinstance(src, str):
            # Resolve registry references
            if src.startswith('@'):
                src = src[1:]
                if src in registry.get_all():
                    obj = registry.get(src)
                else:
                    raise ValueError(
                        f"Can't find {src} in the custom function registry, It may be misspelled or missing "
                        "if you have incorrect optional dependencies installed. Make sure the loading environment"
                        " is the same as the saving environment. For more information, check the Installation "
                        "documentation at "
                        "https://docs.seldon.io/projects/alibi-detect/en/stable/overview/getting_started.html."
                    )
                logger.info('Successfully resolved registry entry {}'.format(src))

            # Resolve dill or numpy file references
            elif Path(src).is_file():
                if Path(src).suffix == '.dill':
                    obj = dill.load(open(src, 'rb'))
                if Path(src).suffix == '.npy':
                    obj = np.load(src, allow_pickle=enable_unsafe_loading)

        # Resolve artefact dicts
        elif isinstance(src, dict):
            backend = cfg.get('backend', Framework.TENSORFLOW)
            if key[-1] in ('model', 'proj'):
                obj = _load_model_config(src)
            elif key[-1] == 'embedding':
                obj = _load_embedding_config(src)
            elif key[-1] == 'tokenizer':
                obj = _load_tokenizer_config(src)
            elif key[-1] == 'optimizer':
                obj = _load_optimizer_config(src, backend)
            elif key[-1] == 'preprocess_fn':
                obj = _load_preprocess_config(src)
            elif key[-1] in ('kernel', 'x_kernel', 'c_kernel'):
                obj = _load_kernel_config(src, backend)

        # Put the resolved function into the cfg dict
        if obj is not None:
            _set_nested_value(cfg, key, obj)

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
        if isinstance(v == orig, bool) and v == orig:
            cfg[k] = new
        elif isinstance(v, dict):
            _replace(v, orig, new)
    return cfg


def _prepend_cfg_filepaths(cfg: dict, prepend_dir: Path):
    """
    Recursively traverse through a nested dictionary and prepend a directory to any filepaths.

    Parameters
    ----------
    cfg
        The dictionary.
    prepend_dir
        The filepath to prepend to any filepaths in the dictionary.

    Returns
    -------
    The updated config dictionary.
    """
    for k, v in cfg.items():
        if isinstance(v, str):
            v = prepend_dir.joinpath(Path(v))
            if v.is_file() or v.is_dir():  # Update if prepending config_dir made config value a real filepath
                cfg[k] = str(v)
        elif isinstance(v, dict):
            _prepend_cfg_filepaths(v, prepend_dir)
