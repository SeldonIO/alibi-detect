from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


class DomainClf(ABC):
    """ 
    Instantiate a domain classifier that takes conditioning variables and their domain
    as input and returns propensity scores (probs of being test instances) as output.
    Should encapsulate logic to ensure an instance's propensity score doesn't depend on its
    own domain (e.g. by using out-of-bag preds or cross validation).
    """
    @abstractmethod
    def __init__(self, *args, **kwargs: dict):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, c_all: np.ndarray, bools: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class SVCDomainClf(DomainClf):
    """
    This fits a SVC on all of the data and then maps scores (that optimise hinge loss) onto
    probabilities using logistic regression. The hope is that by only using 2 parameters to optimise
    the objective of interest prevents overfitts. Empirically this seems to be effective.
    """
    def __init__(self,
        c_kernel: Callable,
        calibrate: bool = True,
        cal_method: str = 'sigmoid',
        clf_kwargs: dict = None):
        self.c_kernel = c_kernel
        self.calibrate = calibrate
        self.cal_method = cal_method
        clf_kwargs = clf_kwargs or {}
        self.clf = SVC(kernel=self.c_kernel, **clf_kwargs)

    def __call__(self,  c_all: np.ndarray,  bools: np.ndarray) -> np.ndarray:
        self.clf.fit(c_all, bools)
        clf_cal = self.clf
        if self.calibrate:
            clf_cal = CalibratedClassifierCV(self.clf, method=self.cal_method, cv='prefit')
            clf_cal.fit(c_all, bools)
        prop_scores = clf_cal.predict_proba(c_all)[:,1]
        return prop_scores
