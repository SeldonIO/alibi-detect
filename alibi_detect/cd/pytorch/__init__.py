from alibi_detect.utils.missing_optional_dependency import import_optional

HiddenOutput, preprocess_drift = import_optional(
    'alibi_detect.cd.pytorch.preprocess',
    names=['HiddenOutput', 'preprocess_drift'])

__all__ = [
    "HiddenOutput",
    "preprocess_drift"
]
