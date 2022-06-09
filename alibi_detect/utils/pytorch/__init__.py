from alibi_detect.utils.missing_optional_dependency import import_optional


TorchDataset = import_optional(
    'alibi_detect.utils.pytorch.data',
    names=['TorchDataset']
)

mmd2, mmd2_from_kernel_matrix, squared_pairwise_distance, permed_lsdds, batch_compute_kernel_matrix = import_optional(
    'alibi_detect.utils.pytorch.distance',
    names=['mmd2', 'mmd2_from_kernel_matrix', 'squared_pairwise_distance',
           'permed_lsdds', 'batch_compute_kernel_matrix']
)

GaussianRBF, DeepKernel = import_optional(
    'alibi_detect.utils.pytorch.kernels',
    names=['GaussianRBF', 'DeepKernel']
)

predict_batch, predict_batch_transformer = import_optional(
    'alibi_detect.utils.pytorch.prediction',
    names=['predict_batch', 'predict_batch_transformer']
)

zero_diag, quantile = import_optional(
    'alibi_detect.utils.pytorch.misc',
    names=['zero_diag', 'quantile']
)


__all__ = [
    "batch_compute_kernel_matrix",
    "mmd2",
    "mmd2_from_kernel_matrix",
    "squared_pairwise_distance",
    "GaussianRBF",
    "DeepKernel",
    "permed_lsdds",
    "predict_batch",
    "predict_batch_transformer",
    "quantile",
    "zero_diag",
    "TorchDataset"
]
