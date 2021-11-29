import catalogue
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

if has_tensorflow:
    from alibi_detect.utils.tensorflow.kernels import GaussianRBF as GaussianRBF_tf
    from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
    from alibi_detect.utils.tensorflow.data import TFDataset as TFDataset_tf

if has_pytorch:
    from alibi_detect.utils.pytorch.kernels import GaussianRBF as GaussianRBF_torch
    from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_torch

# Create registry
registry = catalogue.create("alibi_detect", "registry")

# Register alibi-detect classes/functions
if has_tensorflow:
    registry.register('utils.tensorflow.kernels.GaussianRBF', func=GaussianRBF_tf)
    registry.register('cd.tensorflow.preprocess.preprocess_drift', func=preprocess_drift_tf)
    registry.register('alibi_detect.utils.tensorflow.data.TFDataset', func=TFDataset_tf)

if has_pytorch:
    registry.register('utils.pytorch.kernels.GaussianRBF', func=GaussianRBF_torch)
    registry.register('utils.pytorch.preprocess.preprocess_drift', func=preprocess_drift_torch)
