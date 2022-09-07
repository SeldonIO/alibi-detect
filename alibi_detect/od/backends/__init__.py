from alibi_detect.utils.keops import GaussianRBF
from alibi_detect.utils.missing_optional_dependency import import_optional

KNNKeops = import_optional('alibi_detect.od.backends.keops.knn', ['KNNKeops'])
GaussianRBF = import_optional('alibi_detect.od.backends.torch.kernels', ['GaussianRBF'])
KNNTorch = import_optional('alibi_detect.od.backends.torch.knn', ['KNNTorch'])