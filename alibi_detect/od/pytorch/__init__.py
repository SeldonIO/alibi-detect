from alibi_detect.utils.missing_optional_dependency import import_optional

KNNTorch = import_optional('alibi_detect.od.pytorch.knn', ['KNNTorch'])
MahalanobisTorch = import_optional('alibi_detect.od.pytorch.mahalanobis', ['MahalanobisTorch'])
Accumulator = import_optional('alibi_detect.od.pytorch.ensemble', ['Accumulator'])

to_numpy = import_optional('alibi_detect.od.pytorch.base', ['to_numpy'])