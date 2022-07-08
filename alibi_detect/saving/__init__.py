from alibi_detect.utils.missing_optional_dependency import import_optional

load_detector, read_config, resolve_config = import_optional(
    'alibi_detect.saving.loading',
    names=['load_detector', 'read_config', 'resolve_config']
)

registry = import_optional(
    'alibi_detect.saving.registry',
    names=['registry']
)

save_detector, write_config = import_optional(
    'alibi_detect.saving.saving',
    names=['save_detector', 'write_config']
)

validate_config = import_optional(
    'alibi_detect.saving.validate',
    names=['validate_config']
)

__all__ = [
    "save_detector",
    "write_config",
    "load_detector",
    "read_config",
    "resolve_config",
    "validate_config",
    "registry"
]
