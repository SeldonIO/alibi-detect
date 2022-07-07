from alibi_detect.utils.missing_optional_dependency import import_optional


AE, AEGMM, VAE, VAEGMM, Seq2Seq, eucl_cosim_features = import_optional(
    'alibi_detect.models.tensorflow.autoencoder',
    names=['AE', 'AEGMM', 'VAE', 'VAEGMM', 'Seq2Seq', 'eucl_cosim_features'])

TransformerEmbedding = import_optional(
    'alibi_detect.models.tensorflow.embedding',
    names=['TransformerEmbedding'])

PixelCNN = import_optional(
    'alibi_detect.models.tensorflow.pixelcnn',
    names=['PixelCNN'])

resnet, scale_by_instance = import_optional(
    'alibi_detect.models.tensorflow.resnet',
    names=['resnet', 'scale_by_instance'])

trainer = import_optional(
    'alibi_detect.models.tensorflow.trainer',
    names=['trainer'])

loss_aegmm, loss_adv_ae, loss_distillation, elbo, loss_vaegmm = import_optional(
    'alibi_detect.models.tensorflow.losses',
    names=['loss_aegmm', 'loss_adv_ae', 'loss_distillation', 'elbo', 'loss_vaegmm']
)


__all__ = [
    "AE",
    "AEGMM",
    "Seq2Seq",
    "VAE",
    "VAEGMM",
    "resnet",
    "scale_by_instance",
    "PixelCNN",
    "TransformerEmbedding",
    "trainer",
    "eucl_cosim_features",
    "elbo",
    "loss_aegmm",
    "loss_vaegmm",
    "loss_adv_ae",
    "loss_distillation"
]
