"""Typing and validation

This module contains list of detectors for which save and load functionality is supported. As well as this a
ConfigDetector class that is used to type the save and load

"""


import typing
from typing import Protocol, Dict

ALLOWED_DETECTORS = [
    'AdversarialAE',
    'ChiSquareDrift',
    'ClassifierDrift',
    'IForest',
    'KSDrift',
    'LLR',
    'Mahalanobis',
    'MMDDrift',
    'LSDDDrift',
    'ModelDistillation',
    'OutlierAE',
    'OutlierAEGMM',
    'OutlierProphet',
    'OutlierSeq2Seq',
    'OutlierVAE',
    'OutlierVAEGMM',
    'SpectralResidual',
    'TabularDrift',
    'CVMDrift',
    'FETDrift',
    'SpotTheDiffDrift',
    'ClassifierUncertaintyDrift',
    'RegressorUncertaintyDrift',
    'LearnedKernelDrift',
    'ContextMMDDrift',
    'MMDDriftTF',  # TODO - remove when legacy loading removed
    'ClassifierDriftTF',  # TODO - remove when legacy loading removed
    'MMDDriftOnline',
    'LSDDDriftOnline',
    'CVMDriftOnline',
    'FETDriftOnline'
]


@typing.runtime_checkable
class ConfigurableDetector(Protocol):
    meta: Dict

    def get_config(self):
        ...
