from abc import ABC

from alibi_detect.base import BaseDetector, DriftConfigMixin

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


class ConfigDetector(BaseDetector, DriftConfigMixin, ABC):
    pass
