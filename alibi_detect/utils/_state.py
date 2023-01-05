"""
Submodule to handle saving and loading of detector state dictionaries.

TODO: This submodule will eventually be moved to alibi_detect.saving, however this will require legacy save/load
 support to be refactored or removed, so that the detectors imported in saving._tensorflow.saving/loading do not cause
 circular dependency issues.
"""
import logging

import numpy as np
from pathlib import Path
from alibi_detect.utils.pytorch import _save_state_dict as _save_state_dict_pt, \
    _load_state_dict as _load_state_dict_pt

from alibi_detect.base import BaseDetector

logger = logging.getLogger(__name__)


def save_state_dict(detector: BaseDetector, keys: tuple, filepath: Path):
    """
    Utility function to save a detector's state dictionary to a filepath.

    Parameters
    ----------
    detector
        The detector to extract state attributes from.
    keys
        Tuple of state dict keys to populate dictionary with.
    filepath
        The file to save state dictionary to.
    """
    # Construct state dictionary
    state_dict = {key: getattr(detector, key, None) for key in keys}
    # Save to disk
    if filepath.suffix == '.pt':
        _save_state_dict_pt(state_dict, filepath)
    else:
        np.savez(filepath, **state_dict)


def load_state_dict(detector: BaseDetector, filepath: Path, raise_error: bool = True):
    """
    Utility function to load a detector's state dictionary from a filepath, and update the detectors attributes with
    the values in the state dictionary.

    Parameters
    ----------
     detector
        The detector to update.
    filepath
        File to load state dictionary from.
    raise_error
        Whether to raise an error if a file is not found at `filepath`. Otherwise, raise a warning and skip loading.

    Returns
    -------
    None. The detector is updated inplace.
    """
    if filepath.is_file():
        if filepath.suffix == '.pt':
            state_dict = _load_state_dict_pt(filepath)
        else:
            state_dict = np.load(str(filepath))
        for key, value in state_dict.items():
            setattr(detector, key, value)
    else:
        if raise_error:
            raise FileNotFoundError('State file not found at {}.'.format(filepath))
        else:
            logger.warning('State file not found at {}. Skipping loading of state.'.format(filepath))
