from alibi_detect.utils.saving import load_model, load_tokenizer, load_embedding
import numpy as np
import logging
import os
from pathlib import Path
from typing import Callable, Union, Type, Optional
from ruamel.yaml import YAML
from importlib import import_module
from pydantic import ValidationError
import dill
import catalogue

logger = logging.getLogger(__name__)
custom_artefact = catalogue.create("alibi_detect", "custom_artefact")

# List of absolute paths of nested dict fields we want to resolve.
# NOTE: We could instead just allow any fields (i.e. search entire cfg dict for dict values beginning with @)
FIELDS_TO_RESOLVE = [
    ['x_ref'],
    ['preprocess', 'preprocess_fn'],
    ['preprocess', 'kwargs', 'model'],
    ['preprocess', 'kwargs', 'embedding'],
    ['preprocess', 'kwargs', 'tokenizer']
]


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


def instantiate_class(module: str, name: str, *args, **kwargs) -> Type[Callable]:
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


def resolve_cfg(cfg: dict, verbose: Optional[bool] = False) -> dict:
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
                obj = load_model(src, detector_name=cfg['detector']['type'],
                                 backend=backend, verbose=verbose)
            elif key[-1] == 'embedding':
                obj = load_embedding(src, backend=backend, verbose=verbose)  # TODO
            elif key[-1] == 'tokenizer':
                obj = load_tokenizer(src, backend=backend, verbose=verbose)  # TODO
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
