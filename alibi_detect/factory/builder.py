from alibi_detect.base import BaseDetector
from .utils import read_detector_config
from .detectors import load_detector
from .preprocess import load_preprocessor
from typing import Union, Type
import numpy as np
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


# TODO - could have this integrated into a class as a .from_config() method? Can't think of many reasons for this
#  unless we want to do stuff like combine multiple detectors into one
# TODO - atm this takes in a str referring to config file. Think more about this, e.g. taking in config dict directly.
def DetectorFactory(x_ref: Union[np.ndarray, list], config_file: str) -> BaseDetector:
    # Load the config file
    # NOTE - atm all registry and uri entries are resolved inb load_model(), load_tokenizer() etc. We could make this
    #  cleaner by resolving registry or uri entries in below function instead (i.e. see alibi-explain-factory-config).
    #  However, left as is for now to allow for more granularity i.e. registering custom models separately to
    #  custom tokenizers, layers etc. Can consolidate if not needed...
    cfg = read_detector_config(config_file)
    cfg_orig = deepcopy(cfg)
    backend = cfg.pop('backend', 'tensorflow')

    # Load preprocessor if specified
    if 'preprocess' in cfg:
        preprocessor_cfg = cfg.pop('preprocess')
        preprocess_fn = load_preprocessor(preprocessor_cfg, backend=backend)
    else:
        preprocess_fn = None

    # Load detector
    if 'detector' in cfg:
        detector_cfg = cfg.pop('detector')
        detector = load_detector(x_ref, detector_cfg, preprocess_fn=preprocess_fn, backend=backend)
    else:
        raise ValueError("Config file must contain a 'detector' key.")

    # Update metadata
    detector.meta.update({'config': cfg_orig, 'config_file': config_file})

    return detector
