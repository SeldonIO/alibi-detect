from alibi_detect.utils.missing_optional_dependency import import_optional

KnnKeops = import_optional('alibi_detect.od.backends.keops.knn', ['KnnKeops'])
KnnTorch = import_optional('alibi_detect.od.backends.torch.knn', ['KnnTorch'])