import dill
import toml
import numpy as np
from functools import partial
import logging
import os
from pathlib import Path
from transformers import PreTrainedTokenizerBase
from typing import Callable, Optional, Tuple, Union, get_args
from alibi_detect.saving.loading import Detectors, _replace
from alibi_detect.saving.schemas import SupportedModels
from alibi_detect.saving.registry import registry
from alibi_detect.saving.tensorflow._saving import save_model_config as save_model_config_tf, save_detector_legacy


# do not extend pickle dispatch table so as not to change pickle behaviour
dill.extend(use_dill=False)

logger = logging.getLogger(__name__)


def save_detector(detector: Detectors, filepath: Union[str, os.PathLike], legacy: bool = False) -> None:
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
    """
    if 'backend' in list(detector.meta.keys()) and detector.meta['backend'] in ['pytorch', 'sklearn']:
        raise NotImplementedError('Saving detectors with PyTorch or sklearn backend is not yet supported.')

    detector_name = detector.__class__.__name__
    if detector_name not in [detector.__name__ for detector in get_args(Detectors)]:
        raise ValueError('{} is not supported by `save_detector`.'.format(detector_name))

    # check if path exists
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    # If a drift detector, wrap drift detector save method
    if hasattr(detector, 'get_config') and not legacy:
        _save_detector_config(detector, filepath)

    # Otherwise, save via the previous meta and state_dict approach
    else:
        save_detector_legacy(detector, filepath)

    logger.info('finished saving.')


# TODO - eventually this will become save_detector (once outlier and adversarial updated to save via config.toml)
def _save_detector_config(detector: Detectors, filepath: Union[str, os.PathLike]):
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
    cfg = getattr(detector, 'get_config', None)
    if cfg is not None:
        cfg = cfg()  # TODO - can just do detector.get_config() once all Detectors have a .get_config()
    else:
        raise ValueError('{} is not supported by `save_detector`.'.format(detector_name))

    # Save x_ref
    save_path = filepath.joinpath('x_ref.npy')
    np.save(str(save_path), cfg['x_ref'])
    cfg.update({'x_ref': 'x_ref.npy'})

    # Save preprocess_fn
    preprocess_fn = cfg.get('preprocess_fn', None)
    if preprocess_fn is not None:
        logger.info('Saving the preprocess_fn function.')
        preprocess_cfg = _save_preprocess(preprocess_fn, backend, cfg['input_shape'], filepath)
        cfg['preprocess_fn'] = preprocess_cfg

    # Serialize kernel
    kernel = cfg.get('kernel', None)
    if kernel is not None:
        device = getattr(detector, 'device', None)
        device = device.type if device is not None else None
        cfg['kernel'] = _save_kernel(kernel, filepath, device=device)
        if isinstance(kernel, dict):  # serialise proj from DeepKernel
            cfg['kernel']['proj'], _ = _save_model_config(kernel['proj'], base_path=filepath,
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
        dataset_cfg, dataset_kwargs = _serialize_function(dataset, filepath, Path('dataset'))
        cfg.update({'dataset': dataset_cfg})
        if len(dataset_kwargs) != 0:
            cfg['dataset']['kwargs'] = dataset_kwargs

    # Serialize reg_loss_fn
    reg_loss_fn = cfg.get('reg_loss_fn', None)
    if reg_loss_fn is not None:
        reg_loss_fn_cfg, _ = _serialize_function(reg_loss_fn, filepath, Path('reg_loss_fn'))
        cfg['reg_loss_fn'] = reg_loss_fn_cfg

    # Save initial_diffs
    initial_diffs = cfg.get('initial_diffs', None)
    if initial_diffs is not None:
        save_path = filepath.joinpath('initial_diffs.npy')
        np.save(str(save_path), initial_diffs)
        cfg.update({'initial_diffs': 'initial_diffs.npy'})

    # Save config
    write_config(cfg, filepath)


def write_config(cfg: dict, filepath: Union[str, os.PathLike]) -> dict:
    """
    Save an unresolved detector config dict to a TOML file.

    Parameters
    ----------
    cfg
        Unresolved detector config dict.
    filepath
        Filepath to directory to save 'config.toml' file in.

    Returns
    -------
    The unresolved config dict., with paths converted to string and None converted to "None".
    """
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)
    cfg = _path2str(cfg)
    cfg = _replace(cfg, None, "None")  # Note: None replaced with "None" as None/null not valid TOML
    logger.info('Writing config to {}'.format(filepath.joinpath('config.toml')))
    with open(filepath.joinpath('config.toml'), 'w') as f:
        toml.dump(cfg, f, encoder=toml.TomlNumpyEncoder())  # type: ignore[call-arg, attr-defined]
    return cfg


def _save_preprocess(preprocess_fn: Callable,
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
    func, func_kwargs = _serialize_function(preprocess_fn, filepath, local_path.joinpath('function'))
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
            cfg_token = _save_tokenizer(v, filepath, local_path)
            kwargs.update({k: cfg_token})

        # Arbitrary function
        elif callable(v):
            with open(filepath.joinpath(k + '.dill'), 'wb') as f:
                dill.dump(v, f)
            kwargs.update({k: local_path.joinpath(k + '.dill')})

        # Put remaining kwargs directly into cfg
        else:
            kwargs.update({k: v})

    if 'preprocess_drift' in func:
        preprocess_cfg.update(kwargs)
    else:
        kwargs.update({'kwargs': kwargs})

    return preprocess_cfg


def _serialize_function(func: Callable, base_path: Path,
                        local_path: Path = Path('function')) -> Tuple[str, dict]:
    """
    Serializes a generic function. If the function is in the object registry, the registry str is returned.
    The function is saved to dill, and if wrapped in a functools.partial, the kwargs are returned.

    Parameters
    ----------
    func
        The function to serialize.
    base_path
        Base directory to save in.
    local_path
        A local (relative) filepath to append to base_path. Default is 'function/'.

    Returns
    -------
    Tuple containing a string referencing the save filepath and a dict of kwargs.
    """
    # If a partial, save function and kwargs
    if isinstance(func, partial):
        kwargs = func.keywords
        func = func.func
    else:
        kwargs = {}

    # If a registered function, save registry string
    keys = [k for k, v in registry.get_all().items() if func == v]
    registry_str = keys[0] if len(keys) == 1 else None
    if registry_str is not None:  # alibi-detect registered function
        src = '@' + registry_str

    # Otherwise, save as dill
    else:
        # create folder to save model in
        filepath = base_path.joinpath(local_path)
        if not filepath.parent.is_dir():
            logger.warning('Directory {} does not exist and is now created.'.format(filepath.parent))
            filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.info('Saving function to {}.'.format(filepath.with_suffix('.dill')))
        with open(filepath.with_suffix('.dill'), 'wb') as f:
            dill.dump(func, f)
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
            cfg.update({k: str(v)})
    return cfg


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


def _save_tokenizer(tokenizer: PreTrainedTokenizerBase,
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


def _save_kernel(kernel: Callable,
                 filepath: Path,
                 device: Optional[str] = None,
                 filename: str = 'kernel.dill') -> dict:
    """
    Function to save kernel. If the kernel is stored in the artefact registry, the registry key (and kwargs) are
    written to config. If the kernel is a generic callable, it is pickled.

    Parameters
    ----------
    kernel
        The kernel to save.
    filepath
        Filepath to save to (if the kernel is a generic callable).
    device
        Device. Only needed if pytorch backend being used.
    filename
        Filename to save to (if the kernel is a generic callable).

    Returns
    -------
    The kernel config dictionary.
    """
    cfg_kernel = {}

    keys = [k for k, v in registry.get_all().items() if type(kernel) == v or kernel == v]
    registry_str = keys[0] if len(keys) == 1 else None
    if registry_str is not None:  # alibi-detect registered kernel
        cfg_kernel.update({'src': '@' + registry_str})

        # kwargs for registered kernel
        sigma = getattr(kernel, 'sigma', None)
        sigma = sigma.cpu() if device == 'cuda' else sigma
        cfg_kernel.update({
            'sigma': sigma.numpy().tolist(),
            'trainable': getattr(kernel, 'trainable', None)
        })

    elif isinstance(kernel, dict):  # DeepKernel config dict
        kernel_a = _save_kernel(kernel['kernel_a'], filepath, device, filename='kernel_a.dill')
        kernel_b = kernel.get('kernel_b')
        if kernel_b is not None:
            kernel_b = _save_kernel(kernel['kernel_b'], filepath, device, filename='kernel_b.dill')
        cfg_kernel.update({
            'kernel_a': kernel_a,
            'kernel_b': kernel_b,
            'proj': kernel['proj'],
            'eps': kernel['eps']
        })

    elif callable(kernel):  # generic callable
        logger.info('Saving kernel to {}.'.format(filepath.joinpath(filename)))
        with open(filepath.joinpath(filename), 'wb') as f:
            dill.dump(kernel, f)
        cfg_kernel.update({'src': filename})

    else:  # kernel could not be saved
        raise ValueError("Could not save kernel. Is it a valid Callable or a alibi-detect registered kernel?")

    return cfg_kernel
