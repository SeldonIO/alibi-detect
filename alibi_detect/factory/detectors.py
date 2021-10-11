from alibi_detect.base import BaseDetector
from alibi_detect.cd import ChiSquareDrift, KSDrift, MMDDrift, TabularDrift, LSDDDrift
from alibi_detect.cd import ClassifierUncertaintyDrift, RegressorUncertaintyDrift
from alibi_detect.cd import ClassifierDrift, LearnedKernelDrift, SpotTheDiffDrift
from alibi_detect.utils.pytorch.kernels import DeepKernel, GaussianRBF
from alibi_detect.factory.models import load_model
from alibi_detect.factory.utils import instantiate_class
from typing import Tuple, Union, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)

Detector = Union[
    ChiSquareDrift,
    ClassifierDrift,
    ClassifierUncertaintyDrift,
    KSDrift,
    LearnedKernelDrift,
    LSDDDrift,
    MMDDrift,
    RegressorUncertaintyDrift,
    SpotTheDiffDrift,
    TabularDrift
]

REQUIRES_BACKEND = [
    ClassifierDrift.__name__,
    ClassifierUncertaintyDrift.__name__,
    LearnedKernelDrift.__name__,
    LSDDDrift.__name__,
    MMDDrift.__name__,
    RegressorUncertaintyDrift.__name__,
    SpotTheDiffDrift.__name__
]


REQUIRES_MODEL = [
    ClassifierUncertaintyDrift.__name__,
    RegressorUncertaintyDrift.__name__
]

OPTIONAL_PREPROCESS = [
    ChiSquareDrift.__name__,
    KSDrift.__name__,
    LSDDDrift.__name__,
    MMDDrift.__name__,
    TabularDrift.__name__
]

LEARNED_DETECTOR = [
    ClassifierDrift.__name__,
    LearnedKernelDrift.__name__,
    SpotTheDiffDrift.__name__
]


def init_detector(x_ref: Union[np.ndarray, list],
                  cfg: dict,
                  preprocessor_fn: Optional[Callable] = None,
                  backend: Optional[str] = 'tensorflow') -> BaseDetector:
    # Note: This is setup for drift detectors only atm, but could be modified for od and ad.
    if 'name' in cfg:
        detector_name = cfg.pop('name')
    else:
        raise ValueError('The detector `name` must be specified.')

    # Process args
    args = [x_ref]
    if detector_name in REQUIRES_MODEL:
        if 'model' in cfg:
            model_cfg = cfg.pop('model')
            model = load_model(model_cfg)
            args.append(model)
        else:
            raise ValueError('A `model` must be specified for the %s detector' % detector_name)
    if detector_name in LEARNED_DETECTOR:
        pass  # TODO

    # Process kwargs
    kwargs = {}
    if detector_name in OPTIONAL_PREPROCESS:
        if preprocessor_fn is not None:
            kwargs.update({'preprocessor_fn': preprocessor_fn})
    # Anything remaining in cfg.items() is assumed to be a kwarg
    for k, v in cfg.items():
        kwargs.update({k: v})

    # Instantiate the detector
    detector = instantiate_class('alibi_detect.cd', detector_name, *args, **kwargs)
    logger.info('Instantiated drift detector %s', detector_name)
    return detector


#def init_learned_drift(data_type: str, args: list, kwargs: dict, device: str = None) -> Tuple[list, dict]:
#    args_tmp, kwargs_tmp = args.copy(), kwargs.copy()  # need fresh models each iteration
#    # get the binary classification model or learnable kernel
#    learned_type = 'model' if hasattr(kwargs_tmp, 'model') else 'kernel'
#    learned_name = kwargs_tmp.pop(learned_type)
#    kwargs_tmp.update({'model_kwargs': kwargs_tmp[f'{learned_type}_kwargs']})  # needed to build model
#    model = get_model(learned_name, 2, pretrained=False, **kwargs_tmp)
#    kwargs_tmp.pop('model_kwargs')
#    if learned_type == 'model':
#        args_tmp += [model]
##    else:
##        kwargs_tmp.pop(f'{learned_type}_kwargs')
##        eps = kwargs_tmp.pop('eps')
##        kernel_b = GaussianRBF(trainable=True) if isinstance(eps, float) else None
##        # TODO - if backend=='torch' device=...
##        args_tmp += [DeepKernel(model, kernel_b=kernel_b, eps=eps).to(device)]
#    if 'optimizer' in list(kwargs_tmp.keys()):
#        kwargs_tmp.update({'optimizer': instantiate_class(kwargs_tmp['optimizer'])})
#    return args_tmp, kwargs_tmp
#
#
#def init_detector(name: str, data_type: Optional[str], args: list, kwargs: dict, device: 'str') -> Detector:
#    if name in LEARNED_DETECTOR:
#        args, kwargs = init_learned_drift(data_type, args, kwargs, device=device)
#    return globals()[name](*args, **kwargs)
#