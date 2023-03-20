from alibi_detect.utils.missing_optional_dependency import import_optional

KNNTorch = import_optional('alibi_detect.od.pytorch.knn', ['KNNTorch'])
MahalanobisTorch = import_optional('alibi_detect.od.pytorch.mahalanobis', ['MahalanobisTorch'])
Ensembler = import_optional('alibi_detect.od.pytorch.ensemble', ['Ensembler'])
