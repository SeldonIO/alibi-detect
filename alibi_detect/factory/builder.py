from alibi_detect.base import BaseDetector
from .utils import read_detector_config, resolve_cfg
from .detectors import load_detector
from .preprocess import load_preprocessor
from typing import Union, Optional
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


def DetectorFactory(cfg: Union[str, dict],
                    verbose: Optional[bool] = False) -> BaseDetector:
    # Parse yaml if needed
    if isinstance(cfg, str):
        config_file = deepcopy(cfg)
        cfg = read_detector_config(config_file)
    else:
        config_file = None

    # Resolve cfg
    cfg.update({'registries': {}})
    cfg = resolve_cfg(cfg, verbose=verbose)

    backend = cfg.setdefault('backend', 'tensorflow')

    # x_ref
    x_ref = cfg['x_ref']
    # TODO - if x_ref is still a str raise ValueError

    # Load preprocessor if specified
    if 'preprocess' in cfg:
        preprocessor_cfg = cfg['preprocess']
        preprocess_fn = load_preprocessor(preprocessor_cfg, backend=backend, verbose=verbose)
    else:
        preprocess_fn = None

    # Load detector
    if 'detector' in cfg:
        detector_cfg = cfg['detector']
        detector = load_detector(x_ref, detector_cfg, preprocess_fn=preprocess_fn, backend=backend)
    else:
        raise ValueError("Config file must contain a 'detector' key.")

    # Update metadata
#    detector.meta.update({'config': cfg, 'config_file': config_file})
    detector.meta.update({'config_file': config_file, 'registries': cfg['registries']})

    return detector
