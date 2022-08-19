from alibi_detect.utils.missing_optional_dependency import import_optional


GaussianRBF = import_optional('alibi_detect.utils.keops.kernels', names=['GaussianRBF'])

__all__ = [
    "GaussianRBF"
]
