from alibi_detect.utils.missing_optional_dependency import import_optional


GaussianRBF, DeepKernel = import_optional(
    'alibi_detect.utils.keops.kernels',
    names=['GaussianRBF', 'DeepKernel']
)

__all__ = [
    "GaussianRBF",
    "DeepKernel"
]
