from alibi_detect.utils.missing_optional_dependency import import_optional

KNNKeops = import_optional('alibi_detect.od.backends.keops.knn', ['KNNKeops'])
KNNTorch = import_optional('alibi_detect.od.backends.torch.knn', ['KNNTorch'])