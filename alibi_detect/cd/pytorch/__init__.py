from alibi_detect.utils.missing_optional_dependency import import_optional

UAE, HiddenOutput, preprocess_drift = import_optional(
    'alibi_detect.cd.pytorch.preprocess',
    names=['UAE', 'HiddenOutput', 'preprocess_drift'])

__all__ = [
    "UAE",
    "HiddenOutput",
    "preprocess_drift"
]
