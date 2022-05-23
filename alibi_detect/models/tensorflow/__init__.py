from alibi_detect.utils.missing_optional_dependency import import_optional


AE, AEGMM, VAE, VAEGMM, Seq2Seq = import_optional(
    'alibi_detect.models.tensorflow.autoencoder',
    names=['AE', 'AEGMM', 'VAE', 'VAEGMM', 'Seq2Seq'])

TransformerEmbedding = import_optional(
    'alibi_detect.models.tensorflow.embedding',
    names=['TransformerEmbedding'])

PixelCNN = import_optional(
    'alibi_detect.models.tensorflow.pixelcnn',
    names=['PixelCNN'])

resnet = import_optional(
    'alibi_detect.models.tensorflow.resnet',
    names=['resnet'])

trainer = import_optional(
    'alibi_detect.models.tensorflow.trainer',
    names=['trainer'])


__all__ = [
    "AE",
    "AEGMM",
    "Seq2Seq",
    "VAE",
    "VAEGMM",
    "resnet",
    "PixelCNN",
    "TransformerEmbedding",
    "trainer"
]
