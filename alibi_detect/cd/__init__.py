from alibi_detect.utils.missing_optional_dependency import import_optional

from .chisquare import ChiSquareDrift
from .classifier import ClassifierDrift
from .ks import KSDrift
from .learned_kernel import LearnedKernelDrift
from .lsdd import LSDDDrift
from .lsdd_online import LSDDDriftOnline
from .spot_the_diff import SpotTheDiffDrift
from .mmd import MMDDrift
from .mmd_online import MMDDriftOnline
from .model_uncertainty import ClassifierUncertaintyDrift, RegressorUncertaintyDrift
from .tabular import TabularDrift
from .cvm import CVMDrift
from .fet import FETDrift
from .context_aware import ContextMMDDrift

CVMDriftOnline = import_optional('alibi_detect.cd.cvm_online', names=['CVMDriftOnline'])
FETDriftOnline = import_optional('alibi_detect.cd.fet_online', names=['FETDriftOnline'])


__all__ = [
    "ChiSquareDrift",
    "ClassifierDrift",
    "KSDrift",
    "LearnedKernelDrift",
    "LSDDDrift",
    "LSDDDriftOnline",
    "MMDDrift",
    "MMDDriftOnline",
    "TabularDrift",
    "ClassifierUncertaintyDrift",
    "RegressorUncertaintyDrift",
    "SpotTheDiffDrift",
    "CVMDrift",
    "CVMDriftOnline",
    "FETDrift",
    "FETDriftOnline",
    "ContextMMDDrift"
]
