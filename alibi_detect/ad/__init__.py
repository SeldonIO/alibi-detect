from alibi_detect.utils.missing_optional_dependency import import_optional

AdversarialAE = import_optional('alibi_detect.ad.adversarialae', names=['AdversarialAE'])
ModelDistillation = import_optional('alibi_detect.ad.model_distillation', names=['ModelDistillation'])

__all__ = [
    "AdversarialAE",
    "ModelDistillation"
]
