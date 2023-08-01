import os
from pathlib import Path
import logging
from abc import ABC
from typing import Union, Tuple
import numpy as np
from alibi_detect.utils.frameworks import Framework
from alibi_detect.utils.state._pytorch import save_state_dict as _save_state_dict_pt, \
    load_state_dict as _load_state_dict_pt

logger = logging.getLogger(__name__)


class StateMixin(ABC):
    """
    Utility class that provides methods to save and load stateful attributes to disk.
    """
    t: int
    online_state_keys: Tuple[str, ...]

    def _set_state_dir(self, dirpath: Union[str, os.PathLike]):
        """
        Set the directory path to store state in, and create an empty directory if it doesn't already exist.

        Parameters
        ----------
        dirpath
            The directory to save state file inside.
        """
        self.state_dir = Path(dirpath)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, filepath: Union[str, os.PathLike]):
        """
        Save a detector's state to disk in order to generate a checkpoint.

        Parameters
        ----------
        filepath
            The directory to save state to.
        """
        self._set_state_dir(filepath)
        suffix = '.pt' if hasattr(self, 'backend') and self.backend == Framework.PYTORCH else '.npz'
        _save_state_dict(self, self.online_state_keys, self.state_dir.joinpath('state' + suffix))
        logger.info('Saved state for t={} to {}'.format(self.t, self.state_dir))

    def load_state(self, filepath: Union[str, os.PathLike]):
        """
        Load the detector's state from disk, in order to restart from a checkpoint previously generated with
        `save_state`.

        Parameters
        ----------
        filepath
            The directory to load state from.
        """
        self._set_state_dir(filepath)
        suffix = '.pt' if hasattr(self, 'backend') and self.backend == Framework.PYTORCH else '.npz'
        _load_state_dict(self, self.state_dir.joinpath('state' + suffix), raise_error=True)
        logger.info('State loaded for t={} from {}'.format(self.t, self.state_dir))


def _save_state_dict(detector: StateMixin, keys: tuple, filepath: Path):
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


def _load_state_dict(detector: StateMixin, filepath: Path, raise_error: bool = True):
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
