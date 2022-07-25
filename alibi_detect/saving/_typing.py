"""Typing constructs for saving and loading functionality

List of detectors that are valid for saving and loading either via the legacy methods or the new config driven
functionality"""

VALID_DETECTORS = [
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
