import catalogue
from alibi_detect.utils.tensorflow.kernels import GaussianRBF as GaussianRBF_tf
from alibi_detect.utils.pytorch.kernels import GaussianRBF as GaussianRBF_torch
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_torch

# Create registry
registry = catalogue.create("alibi_detect", "registry")

# Register alibi-detect classes/functions
registry.register('GaussianRBF_tf', func=GaussianRBF_tf)
registry.register('GaussianRBF_torch', func=GaussianRBF_torch)
registry.register('preprocess_drift_torch', func=preprocess_drift_torch)
registry.register('preprocess_drift_tf', func=preprocess_drift_tf)
