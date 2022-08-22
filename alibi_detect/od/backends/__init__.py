from alibi_detect.utils.missing_optional_dependency import import_optional

KnnTorch = import_optional('alibi_detect.od.backends.torch.knn', ['KnnTorch'])