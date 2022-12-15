"""
Submodule to handle saving and loading of detector state dictionaries when the dictionaries contain `torch.Tensor`'s.

TODO: This submodule will eventually be moved to alibi_detect.saving.pytorch, however this will require legacy
 save/load support to be refactored or removed, so that the detectors imported in saving._tensorflow.saving/loading
 do not cause circular dependency issues.
"""
from pathlib import Path
import torch


def save_state_dict(state_dict: dict, filepath: Path):
    """
    Utility function to save a detector's state dictionary to a filepath using `torch.save`.

    Parameters
    ----------
    state_dict
        The state dictionary to save.
    filepath
        Directory to save state dictionary to.
    """
    # Save to disk
    torch.save(state_dict, filepath)


def load_state_dict(filepath: Path) -> dict:
    """
    Utility function to load a detector's state dictionary from a filepath with `torch.load`.

    Parameters
    ----------
    filepath
        Directory to load state dictionary from.

    Returns
    -------
    The loaded state dictionary.
    """
    return torch.load(filepath)
