from alibi_detect.utils.missing_optional_dependency import import_optional

load_detector_legacy, load_kernel_config_tf, load_embedding_tf, load_model_tf, load_optimizer_tf, \
    prep_model_and_emb_tf = import_optional(
        'alibi_detect.saving._tensorflow.loading',
        names=['load_detector_legacy',
               'load_kernel_config',
               'load_embedding',
               'load_model',
               'load_optimizer',
               'prep_model_and_emb'])

save_detector_legacy, save_model_config_tf, save_optimizer_config_tf = import_optional(
    'alibi_detect.saving._tensorflow.saving',
    names=['save_detector_legacy', 'save_model_config', 'save_optimizer_config']
)

get_tf_dtype = import_optional(
    'alibi_detect.saving._tensorflow.conversions',
    names=['get_tf_dtype']
)

__all__ = [
    "load_detector_legacy",
    "load_kernel_config_tf",
    "load_embedding_tf",
    "load_model_tf",
    "load_optimizer_tf",
    "prep_model_and_emb_tf",
    "save_detector_legacy",
    "save_model_config_tf",
    "save_optimizer_config_tf",
    "get_tf_dtype"
]
