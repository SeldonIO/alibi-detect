from alibi_detect.utils.missing_optional_dependency import import_optional

KNNTorch = import_optional('alibi_detect.od.pytorch.knn', ['KNNTorch'])
Ensembler = import_optional('alibi_detect.od.pytorch.ensemble', ['Ensembler'])

to_numpy = import_optional('alibi_detect.od.pytorch.base', ['to_numpy'])
