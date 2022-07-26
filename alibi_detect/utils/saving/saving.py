# This submodule provides a link for the legacy alibi_detect.utils.saving location of load_detector and save_detector.
# TODO: Remove in future
from alibi_detect.saving import load_detector as _load_detector, save_detector as _save_detector
from alibi_detect.base import ConfigurableDetector, Detector

from typing import Union
import warnings
import os


def save_detector(
        detector: Union[Detector, ConfigurableDetector],
        filepath: Union[str, os.PathLike], legacy: bool = False) -> None:
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
    warnings.warn("This function has been moved to alibi_detect.saving.save_detector()."
                  "This legacy link will be removed in a future version", DeprecationWarning)
    return _save_detector(detector, filepath, legacy)


def load_detector(filepath: Union[str, os.PathLike], **kwargs) -> Union[Detector, ConfigurableDetector]:
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
    warnings.warn("This function has been moved to alibi_detect.saving.load_detector()."
                  "This legacy link will be removed in a future version", DeprecationWarning)
    return _load_detector(filepath, **kwargs)
