from alibi_detect.version import __version__, __config_spec__
from alibi_detect.utils.schemas import DETECTOR_CONFIGS, DETECTOR_CONFIGS_RESOLVED  # type: ignore[attr-defined]
import warnings


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
    meta = cfg.pop('meta', {})

    # Validate detector specific config
    if detector_name in DETECTOR_CONFIGS.keys():
        if resolved:
            cfg = DETECTOR_CONFIGS_RESOLVED[detector_name](**cfg).dict()
        else:
            cfg = DETECTOR_CONFIGS[detector_name](**cfg).dict()
    else:
        raise ValueError(f'Loading the {detector_name} detector from a config.toml is not yet supported.')

    # Raise warning if config file already contains a version_warning
    version_warning = meta.pop('version_warning', False)
    if version_warning:
        warnings.warn('The config file appears to be have been generated from a detector which may have been '
                      'loaded with a version mismatch. This may lead to breaking code or invalid results.')

    # check version
    version = meta.pop('version', None)
    if version is not None and version != __version__:
        warnings.warn(f'Config is from version {version} but current version is '
                      f'{__version__}. This may lead to breaking code or invalid results.')
        cfg['meta'].update({'version_warning': True})

    # Check config specification version
    config_spec = meta.pop('config_spec', None)
    if config_spec is not None and config_spec != __config_spec__:
        warnings.warn(f'Config has specification {version} when the installed '
                      f'alibi-detect version expects specification {__config_spec__}.'
                      'This may lead to breaking code or invalid results.')
        cfg['meta'].update({'version_warning': True})

    return cfg
