import logging
import os
import shutil
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import dill
import numpy as np
import toml
from transformers import PreTrainedTokenizerBase

from alibi_detect.saving._typing import VALID_DETECTORS
from alibi_detect.saving.loading import _replace, validate_config
from alibi_detect.saving.registry import registry
from alibi_detect.saving.schemas import SupportedModels
from alibi_detect.saving.tensorflow import save_detector_legacy, save_model_config_tf
from alibi_detect.base import Detector, ConfigurableDetector

# do not extend pickle dispatch table so as not to change pickle behaviour
dill.extend(use_dill=False)

logger = logging.getLogger(__name__)

X_REF_FILENAME = 'x_ref.npy'
C_REF_FILENAME = 'c_ref.npy'


def save_detector(
        detector: Union[Detector, ConfigurableDetector],
        filepath: Union[str, os.PathLike], legacy: bool = False
        ) -> None:
    """
    Save outlier, drift or adversarial detector.

    Parameters
    ----------
    detector
        Detector object.
    filepath
        Save directory.
    legacy
        Whether to save in the legacy .dill format instead of via a config.toml file. Default is `False`.
        This option will be removed in a future version.
    """
    if legacy:
        warnings.warn('The `legacy` option will be removed in a future version.', DeprecationWarning)

    if 'backend' in list(detector.meta.keys()) and detector.meta['backend'] in ['pytorch', 'sklearn']:
        raise NotImplementedError('Saving detectors with PyTorch or sklearn backend is not yet supported.')

    # TODO: Replace .__args__ w/ typing.get_args() once Python 3.7 dropped (and remove type ignore below)
    detector_name = detector.__class__.__name__
    if detector_name not in [detector for detector in VALID_DETECTORS]:
        raise NotImplementedError(f'{detector_name} is not supported by `save_detector`.')

    # Saving is wrapped in a try, with cleanup in except. To prevent a half-saved detector remaining upon error.
    filepath = Path(filepath)
    try:
        # Create directory if it doesn't exist
        if not filepath.is_dir():
            logger.warning('Directory {} does not exist and is now created.'.format(filepath))
            filepath.mkdir(parents=True, exist_ok=True)

        # If a drift detector, wrap drift detector save method
        if isinstance(detector, ConfigurableDetector) and not legacy:
            _save_detector_config(detector, filepath)

        # Otherwise, save via the previous meta and state_dict approach
        else:
            save_detector_legacy(detector, filepath)

    except Exception as error:
        # Get a list of all existing files in `filepath` (so we know what not to cleanup if an error occurs)
        orig_files = set(filepath.iterdir())
        _cleanup_filepath(orig_files, filepath)
        raise RuntimeError(f'Saving failed. The save directory {filepath} has been cleaned.') from error

    logger.info('finished saving.')


def _cleanup_filepath(orig_files: set, filepath: Path):
    """
    Cleans up the `filepath` directory in the event of a saving failure.

    Parameters
    ----------
    orig_files
        Set of original files (not to delete).
    filepath
        The directory to clean up.
    """
    # Find new files
    new_files = set(filepath.iterdir())
    files_to_rm = new_files - orig_files
    # Delete new files
    for file in files_to_rm:
        if file.is_dir():
            shutil.rmtree(file)
        elif file.is_file():
            file.unlink()

    # Delete filepath directory if it is now empty
    if filepath is not None:
        if not any(filepath.iterdir()):
            filepath.rmdir()


# TODO - eventually this will become save_detector (once outlier and adversarial updated to save via config.toml)
def _save_detector_config(detector: ConfigurableDetector, filepath: Union[str, os.PathLike]):
    """
    Save a drift detector. The detector is saved as a yaml config file. Artefacts such as
    `preprocess_fn`, models, embeddings, tokenizers etc are serialized, and their filepaths are
    added to the config file.

    The detector can be loaded again by passing the resulting config file or filepath to `load_detector`.

    Parameters
    ----------
    detector
        The detector to save.
    filepath
        File path to save serialized artefacts to.
    """
    # Get backend, input_shape and detector_name
    backend = detector.meta.get('backend', 'tensorflow')
    if backend != 'tensorflow':
        raise NotImplementedError("Currently, saving is only supported with backend='tensorflow'.")
    detector_name = detector.__class__.__name__

    # Process file paths
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    # Get the detector config (with artefacts still within it)
    if hasattr(detector, 'get_config'):
        cfg = detector.get_config()  # type: ignore[union-attr]  # TODO - remove once all detectors have get_config
        cfg = validate_config(cfg, resolved=True)
    else:
        raise NotImplementedError(f'{detector_name} does not yet support config.toml based saving.')

    # Save x_ref
    save_path = filepath.joinpath(X_REF_FILENAME)
    np.save(str(save_path), cfg['x_ref'])
    cfg.update({'x_ref': X_REF_FILENAME})

    # Save c_ref
    c_ref = cfg.get('c_ref', None)
    if c_ref is not None:
        save_path = filepath.joinpath(C_REF_FILENAME)
        np.save(str(save_path), cfg['c_ref'])
        cfg.update({'c_ref': C_REF_FILENAME})

    # Save preprocess_fn
    preprocess_fn = cfg.get('preprocess_fn', None)
    if preprocess_fn is not None:
        logger.info('Saving the preprocess_fn function.')
        preprocess_cfg = _save_preprocess_config(preprocess_fn, backend, cfg['input_shape'], filepath)
        cfg['preprocess_fn'] = preprocess_cfg

    # Serialize kernels
    for kernel_str in ('kernel', 'x_kernel', 'c_kernel'):
        kernel = cfg.get(kernel_str, None)
        if kernel is not None:
            cfg[kernel_str] = _save_kernel_config(kernel, filepath, Path(kernel_str))
            if 'proj' in cfg[kernel_str]:  # serialise proj from DeepKernel - do here as need input_shape
                cfg[kernel_str]['proj'], _ = _save_model_config(cfg[kernel_str]['proj'], base_path=filepath,
                                                                input_shape=cfg['input_shape'], backend=backend)

    # ClassifierDrift and SpotTheDiffDrift specific artefacts.
    # Serialize detector model
    model = cfg.get('model', None)
    if model is not None:
        model_cfg, _ = _save_model_config(model, base_path=filepath, input_shape=cfg['input_shape'], backend=backend)
        cfg['model'] = model_cfg

    # Serialize dataset
    dataset = cfg.get('dataset', None)
    if dataset is not None:
        dataset_cfg, dataset_kwargs = _serialize_object(dataset, filepath, Path('dataset'))
        cfg.update({'dataset': dataset_cfg})
        if len(dataset_kwargs) != 0:
            cfg['dataset']['kwargs'] = dataset_kwargs

    # Serialize reg_loss_fn
    reg_loss_fn = cfg.get('reg_loss_fn', None)
    if reg_loss_fn is not None:
        reg_loss_fn_cfg, _ = _serialize_object(reg_loss_fn, filepath, Path('reg_loss_fn'))
        cfg['reg_loss_fn'] = reg_loss_fn_cfg

    # Save initial_diffs
    initial_diffs = cfg.get('initial_diffs', None)
    if initial_diffs is not None:
        save_path = filepath.joinpath('initial_diffs.npy')
        np.save(str(save_path), initial_diffs)
        cfg.update({'initial_diffs': 'initial_diffs.npy'})

    # Save config
    write_config(cfg, filepath)


def write_config(cfg: dict, filepath: Union[str, os.PathLike]):
    """
    Save an unresolved detector config dict to a TOML file.

    Parameters
    ----------
    cfg
        Unresolved detector config dict.
    filepath
        Filepath to directory to save 'config.toml' file in.
    """
    # Create directory if it doesn't exist
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)
    # Convert pathlib.Path's to str's
    cfg = _path2str(cfg)
    # Validate config before final tweaks
    validate_config(cfg)  # Must validate here as replacing None w/ str will break validation
    # Replace None with "None", and dicts with integer keys with str keys
    # TODO: Subject to change depending on toml library updates
    cfg = _replace(cfg, None, "None")  # Note: None replaced with "None" as None/null not valid TOML
    cfg = _int2str_keys(cfg)
    # Write to TOML file
    logger.info('Writing config to {}'.format(filepath.joinpath('config.toml')))
    with open(filepath.joinpath('config.toml'), 'w') as f:
        toml.dump(cfg, f, encoder=toml.TomlNumpyEncoder())  # type: ignore[misc]


def _save_preprocess_config(preprocess_fn: Callable,
                            backend: str,
                            input_shape: Optional[tuple],
                            filepath: Path) -> dict:
    """
    Serializes a drift detectors preprocess_fn. Artefacts are saved to disk, and a config dict containing filepaths
    to the saved artefacts is returned.

    Parameters
    ----------
    preprocess_fn
        The preprocess function to be serialized.
    backend
        Specifies the detectors backend (if it has one). Either `'tensorflow'`, `'pytorch'` or `None`.
    input_shape
        Input shape for a model (if a model exists).
    filepath
        Directory to save serialized artefacts to.

    Returns
    -------
    The config dictionary, containing references to the serialized artefacts. The format if this dict matches that
    of the `preprocess` field in the drift detector specification.
    """
    preprocess_cfg = {}
    local_path = Path('preprocess_fn')

    # Serialize function
    func, func_kwargs = _serialize_object(preprocess_fn, filepath, local_path.joinpath('function'))
    preprocess_cfg.update({'src': func})

    # Process partial function kwargs (if they exist)
    kwargs = {}
    for k, v in func_kwargs.items():
        # Model/embedding
        if isinstance(v, SupportedModels):
            cfg_model, cfg_embed = _save_model_config(v, filepath, input_shape, backend, local_path)
            kwargs.update({k: cfg_model})
            if cfg_embed is not None:
                kwargs.update({'embedding': cfg_embed})

        # Tokenizer
        elif isinstance(v, PreTrainedTokenizerBase):
            cfg_token = _save_tokenizer_config(v, filepath, local_path)
            kwargs.update({k: cfg_token})

        # Arbitrary function
        elif callable(v):
            src, _ = _serialize_object(v, filepath, local_path)
            kwargs.update({k: src})

        # Put remaining kwargs directly into cfg
        else:
            kwargs.update({k: v})

    if 'preprocess_drift' in func:
        preprocess_cfg.update(kwargs)
    else:
        kwargs.update({'kwargs': kwargs})

    return preprocess_cfg


def _serialize_object(obj: Callable, base_path: Path,
                      local_path: Path = Path('.')) -> Tuple[str, dict]:
    """
    Serializes a python object. If the object is in the object registry, the registry str is returned. If not,
    the object is saved to dill, and if wrapped in a functools.partial, the kwargs are returned.

    Parameters
    ----------
    obj
        The object to serialize.
    base_path
        Base directory to save in.
    local_path
        A local (relative) filepath to append to base_path.

    Returns
    -------
    Tuple containing a string referencing the save filepath and a dict of kwargs.
    """
    # If a functools.partial, unpick function and kwargs
    if isinstance(obj, partial):
        kwargs = obj.keywords
        obj = obj.func
    else:
        kwargs = {}

    # If object has been registered, save registry string
    keys = [k for k, v in registry.get_all().items() if obj == v]
    registry_str = keys[0] if len(keys) == 1 else None
    if registry_str is not None:  # alibi-detect registered object
        src = '@' + registry_str

    # Otherwise, save as dill
    else:
        # create folder to save object in
        filepath = base_path.joinpath(local_path)
        if not filepath.parent.is_dir():
            logger.warning('Directory {} does not exist and is now created.'.format(filepath.parent))
            filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.info('Saving object to {}.'.format(filepath.with_suffix('.dill')))
        with open(filepath.with_suffix('.dill'), 'wb') as f:
            dill.dump(obj, f)
        src = str(local_path.with_suffix('.dill'))

    return src, kwargs


def _path2str(cfg: dict, absolute: bool = False) -> dict:
    """
    Private function to traverse a config dict and convert pathlib Path's to strings.

    Parameters
    ----------
    cfg
        The config dict.
    absolute
        Whether to convert to absolute filepaths.

    Returns
    -------
    The converted config dict.
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            _path2str(v, absolute)
        elif isinstance(v, Path):
            if absolute:
                v = v.resolve()
            cfg.update({k: str(v.as_posix())})
    return cfg


def _int2str_keys(dikt: dict) -> dict:
    """
    Private function to traverse a dict and convert any dict's with int keys to str keys (e.g.
    `categories_per_feature` kwarg for `TabularDrift`.

    Parameters
    ----------
    dikt
        The dictionary.

    Returns
    -------
    The converted dictionary.
    """
    dikt_copy = dikt.copy()
    for k, v in dikt.items():
        if isinstance(k, int):
            dikt_copy[str(k)] = dikt[k]
            dikt_copy.pop(k)
        if isinstance(v, dict):
            dikt_copy[k] = _int2str_keys(v)
    return dikt_copy


def _save_model_config(model: Callable,
                       base_path: Path,
                       input_shape: tuple,
                       backend: str,
                       path: Path = Path('.')) -> Tuple[dict, Optional[dict]]:
    """
    Save a model to a config dictionary. When a model has a text embedding model contained within it,
    this is extracted and saved separately.

    Parameters
    ----------
    model
        The model to save.
    base_path
        Base filepath to save to.
    input_shape
        The input dimensions of the model (after the optional embedding has been applied).
    backend
        The backend.
    path
        A local (relative) filepath to append to base_path.

    Returns
    -------
    A tuple containing the model and embedding config dicts.
    """
    if backend == 'tensorflow':
        return save_model_config_tf(model, base_path, input_shape, path)
    else:
        raise NotImplementedError("Saving of pytorch models is not yet implemented.")


def _save_tokenizer_config(tokenizer: PreTrainedTokenizerBase,
                           base_path: Path,
                           path: Path = Path('.')) -> dict:
    """
    Saves HuggingFace tokenizers.

    Parameters
    ----------
    tokenizer
        The tokenizer.
    base_path
        Base filepath to save to.
    path
        A local (relative) filepath to append to base_path.

    Returns
    -------
    The tokenizer config dict.
    """
    # create folder to save model in
    filepath = base_path.joinpath(path).joinpath('tokenizer')
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    cfg_token = {}
    logger.info('Saving tokenizer to {}.'.format(filepath))
    tokenizer.save_pretrained(filepath)
    cfg_token.update({'src': path.joinpath('tokenizer')})
    return cfg_token


def _save_kernel_config(kernel: Callable,
                        base_path: Path,
                        local_path: Path = Path('.')) -> dict:
    """
    Function to save kernel. If the kernel is stored in the artefact registry, the registry key (and kwargs) are
    written to config. If the kernel is a generic callable, it is pickled.

    Parameters
    ----------
    kernel
        The kernel to save.
     base_path
        Base directory to save in.
    local_path
        A local (relative) filepath to append to base_path.

    Returns
    -------
    The kernel config dictionary.
    """
    # if a DeepKernel
    if hasattr(kernel, 'proj'):
        if hasattr(kernel, 'get_config'):
            cfg_kernel = kernel.get_config()  # type: ignore[attr-defined]
        else:
            raise AttributeError("The detector's `kernel` must have a .get_config() method for it to be saved.")
        # Serialize the kernels (if needed)
        kernel_a = cfg_kernel.get('kernel_b')
        kernel_b = cfg_kernel.get('kernel_b')
        if not isinstance(kernel_a, str):
            cfg_kernel['kernel_a'] = _save_kernel_config(cfg_kernel['kernel_a'], base_path, Path('kernel_a'))
        if not isinstance(kernel_b, str) and kernel_b is not None:
            cfg_kernel['kernel_b'] = _save_kernel_config(cfg_kernel['kernel_b'], base_path, Path('kernel_b'))

    # If any other kernel, serialize the class to disk and get config
    else:
        if isinstance(kernel, type):  # if still a class
            kernel_class = kernel
            cfg_kernel = {}
        else:  # if an object
            kernel_class = kernel.__class__
            if hasattr(kernel, 'get_config'):
                cfg_kernel = kernel.get_config()  # type: ignore[attr-defined]
                cfg_kernel['init_sigma_fn'], _ = _serialize_object(cfg_kernel['init_sigma_fn'], base_path,
                                                                   local_path.joinpath('init_sigma_fn'))
            else:
                raise AttributeError("The detector's `kernel` must have a .get_config() method for it to be saved.")
        # Serialize the kernel class
        cfg_kernel['src'], _ = _serialize_object(kernel_class, base_path, local_path.joinpath('kernel'))

    return cfg_kernel
