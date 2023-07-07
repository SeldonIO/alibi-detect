from alibi_detect.utils.missing_optional_dependency import import_optional

KNNTorch = import_optional('alibi_detect.od.pytorch.knn', ['KNNTorch'])
LOFTorch = import_optional('alibi_detect.od.pytorch.lof', ['LOFTorch'])
MahalanobisTorch = import_optional('alibi_detect.od.pytorch.mahalanobis', ['MahalanobisTorch'])
KernelPCATorch, LinearPCATorch = import_optional('alibi_detect.od.pytorch.pca', ['KernelPCATorch', 'LinearPCATorch'])
Ensembler = import_optional('alibi_detect.od.pytorch.ensemble', ['Ensembler'])
GMMTorch = import_optional('alibi_detect.od.pytorch.gmm', ['GMMTorch'])
BgdSVMTorch, SgdSVMTorch = import_optional('alibi_detect.od.pytorch.svm', ['BgdSVMTorch', 'SgdSVMTorch'])
