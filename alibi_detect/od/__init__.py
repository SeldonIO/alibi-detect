from alibi_detect.utils.missing_optional_dependency import import_optional

from .isolationforest import IForest
from .mahalanobis import Mahalanobis
from .prophet import PROPHET_INSTALLED, OutlierProphet
from .sr import SpectralResidual

OutlierAEGMM = import_optional('alibi_detect.od.aegmm', names=['OutlierAEGMM'])
OutlierAE = import_optional('alibi_detect.od.ae', names=['OutlierAE'])
OutlierVAE = import_optional('alibi_detect.od.vae', names=['OutlierVAE'])
OutlierVAEGMM = import_optional('alibi_detect.od.vaegmm', names=['OutlierVAEGMM'])
OutlierSeq2Seq = import_optional('alibi_detect.od.seq2seq', names=['OutlierSeq2Seq'])
LLR = import_optional('alibi_detect.od.llr', names=['LLR'])

__all__ = [
    "OutlierAEGMM",
    "IForest",
    "Mahalanobis",
    "OutlierAE",
    "OutlierVAE",
    "OutlierVAEGMM",
    "OutlierSeq2Seq",
    "SpectralResidual",
    "LLR"
]

if PROPHET_INSTALLED:
    __all__ += ["OutlierProphet"]
