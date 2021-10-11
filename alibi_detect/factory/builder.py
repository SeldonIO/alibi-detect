from alibi_detect.base import BaseDetector
from .utils import read_detector_config
from .detectors import init_detector
from typing import Union, Type
import numpy as np
import logging

logger = logging.getLogger(__name__)


# TODO - could have this integrated into a class as a .from_config() method? Can't think of many reasons for this
#  unless we want to do stuff like combine multiple detectors into one, or do more complex serialization etc
# TODO - atm this takes in a str referring to config file. Think more about this, e.g. taking in config dict directly.
def DetectorFactory(x_ref: Union[np.ndarray, list], config_file: str) -> BaseDetector:
    # Load the config file
    cfg = read_detector_config(config_file)

# Load preprocessor if specified
#    if 'preprocess' in cfg:
#        preprocessor_cfg = cfg.pop('preprocess')
#        preprocessor_fn = init_preprocessor(preprocessor_cfg)

    # Load detector
    if 'detector' in cfg:
        detector_cfg = cfg.pop('detector')
        detector = init_detector(x_ref, detector_cfg)
    else:
        raise ValueError("Config file must contain a 'detector' key.")

    # Update metadata
    detector.meta.update({'config_file': config_file})
    return detector
