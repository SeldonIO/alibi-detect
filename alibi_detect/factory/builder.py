from alibi_detect.base import BaseDetector
from .utils import read_detector_config, resolve_cfg
from .detectors import load_detector
from .preprocess import load_preprocessor
from typing import Union, Optional
import numpy as np
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


# TODO - could have this integrated into a class as a .from_config() method? Can't think of many reasons for this
#  unless we want to do stuff like combine multiple detectors into one? i.e. give DetectorFactory a list of yaml files,
#  specifying multiple detectors. DetectorFactory would have predict() method etc to call all detectors at once...
def DetectorFactory(x_ref: Union[np.ndarray, list],
                    cfg: Union[str, dict],
                    verbose: Optional[bool] = False) -> BaseDetector:
    # Parse yaml if needed
    if isinstance(cfg, str):
        config_file = deepcopy(cfg)
        cfg = read_detector_config(config_file)
    else:
        config_file = None

    # Resolve and validate cfg
    cfg.update({'registries': {}})
    cfg = resolve_cfg(cfg, verbose=verbose)
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

    # Warn about unused fields
    if verbose and len(cfg) > 0:
        logger.warning("DetectorFactory: some `config_file` fields were unused: %s" % cfg.keys())

    return detector
