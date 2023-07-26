from alibi_detect.utils.missing_optional_dependency import import_optional

load_kernel_config_pt, load_embedding_pt, load_model_pt, load_optimizer_pt, \
    prep_model_and_emb_pt = import_optional(
        'alibi_detect.saving._pytorch.loading',
        names=['load_kernel_config',
               'load_embedding',
               'load_model',
               'load_optimizer',
               'prep_model_and_emb'])

save_model_config_pt, save_device_pt = import_optional(
    'alibi_detect.saving._pytorch.saving',
    names=['save_model_config', 'save_device']
)

get_pt_dtype = import_optional(
    'alibi_detect.saving._pytorch.conversions',
    names=['get_pt_dtype']
)

__all__ = [
    "load_kernel_config_pt",
    "load_embedding_pt",
    "load_model_pt",
    "load_optimizer_pt",
    "prep_model_and_emb_pt",
    "save_model_config_pt",
    "get_pt_dtype"
]
