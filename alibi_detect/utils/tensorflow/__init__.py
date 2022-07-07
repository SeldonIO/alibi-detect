from alibi_detect.utils.missing_optional_dependency import import_optional

mmd2, mmd2_from_kernel_matrix, batch_compute_kernel_matrix, relative_euclidean_distance, squared_pairwise_distance, \
    permed_lsdds = import_optional(
        'alibi_detect.utils.tensorflow.distance',
        names=['mmd2', 'mmd2_from_kernel_matrix', 'batch_compute_kernel_matrix', 'relative_euclidean_distance',
               'squared_pairwise_distance', 'permed_lsdds']
    )


GaussianRBF, DeepKernel = import_optional(
    'alibi_detect.utils.tensorflow.kernels',
    names=['GaussianRBF', 'DeepKernel']
)


predict_batch, predict_batch_transformer = import_optional(
    'alibi_detect.utils.tensorflow.prediction',
    names=['predict_batch', 'predict_batch_transformer']
)


zero_diag, quantile, subset_matrix = import_optional(
    'alibi_detect.utils.tensorflow.misc',
    names=['zero_diag', 'quantile', 'subset_matrix']
)


mutate_categorical = import_optional(
    'alibi_detect.utils.tensorflow.perturbation',
    names=['mutate_categorical']
)


TFDataset = import_optional(
    'alibi_detect.utils.tensorflow.data',
    names=['TFDataset']
)


__all__ = [
    "batch_compute_kernel_matrix",
    "mmd2",
    "mmd2_from_kernel_matrix",
    "relative_euclidean_distance",
    "squared_pairwise_distance",
    "GaussianRBF",
    "DeepKernel",
    "permed_lsdds",
    "predict_batch",
    "predict_batch_transformer",
    "quantile",
    "subset_matrix",
    "zero_diag",
    "mutate_categorical",
    "TFDataset"
]
