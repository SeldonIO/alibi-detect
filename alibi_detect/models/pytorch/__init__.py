from alibi_detect.utils.missing_optional_dependency import import_optional

TransformerEmbedding = import_optional(
    'alibi_detect.models.pytorch.embedding',
    names=['TransformerEmbedding'])

trainer = import_optional(
    'alibi_detect.models.pytorch.trainer',
    names=['trainer'])


__all__ = [
    "TransformerEmbedding",
    "trainer"
]
