import warnings

from alibi_detect.saving.schemas import (
    DETECTOR_CONFIGS, DETECTOR_CONFIGS_RESOLVED)
from alibi_detect.version import __version__


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
    # Get detector name and meta
    if 'name' in cfg:
        detector_name = cfg['name']
    else:
        raise ValueError('`name` missing from config.toml.')

    # Validate detector specific config
    if detector_name in DETECTOR_CONFIGS.keys():
        if resolved:
            cfg = DETECTOR_CONFIGS_RESOLVED[detector_name](**cfg).dict()
        else:
            cfg = DETECTOR_CONFIGS[detector_name](**cfg).dict()
    else:
        raise ValueError(f'Loading the {detector_name} detector from a config.toml is not yet supported.')

    # Get meta data
    meta = cfg.get('meta')
    meta = {} if meta is None else meta  # Needed because pydantic sets meta=None if it is missing from the config
    version_warning = meta.get('version_warning', False)
    version = meta.get('version', None)

    # Raise warning if config file already contains a version_warning
    if version_warning:
        warnings.warn('The config file appears to be have been generated from a detector which may have been '
                      'loaded with a version mismatch. This may lead to breaking code or invalid results.')

    # check version
    if version is not None and version != __version__:
        warnings.warn(f'Config is from version {version} but current version is '
                      f'{__version__}. This may lead to breaking code or invalid results.')
        cfg['meta'].update({'version_warning': True})

    return cfg
