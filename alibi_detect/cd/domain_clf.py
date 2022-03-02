from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


class DomainClf(ABC):
    """
    Base class for a domain classifier. Classifiers passed to `domain_clf` in
    :py:class:`~alibi_detect.cd.ContextAwareDrift` are expected to be a subclass of this.

    The classifiers should take conditioning variables `c_all` and their domain `bools` as input (`0` for ref, `1`
    test), and return propensity scores (probability of being test instances) as output. Classifiers should also
    encapsulate logic to ensure an instance's propensity score doesn't depend on its own domain (e.g. by using
    out-of-bag predictions or cross validation).
    """
    @abstractmethod
    def __init__(self, *args, **kwargs: dict):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, c_all: np.ndarray, bools: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class SVCDomainClf(DomainClf):
    def __init__(self,
                 c_kernel: Callable,
                 calibrate: bool = True,
                 cal_method: str = 'sigmoid',
                 clf_kwargs: dict = None):
        """
        A domain classifier using the scikit-learn Support Vector Classifier
        (:py:class:`~sklearn.svm.SVC`). An SVC is fitted on all the
        data, with scores (that optimise hinge loss) mapped onto probabilities using logistic regression.

        Parameters
        ----------
        c_kernel
            Kernel used to pre-compute the kernel matrix from data matrices.
        calibrate
            Whether to perform probability calibration on the SVC model.
        cal_method
            The method to be used to calibrate the detector. This should be a method accepted by the scikit-learn
            :py:class:`~sklearn.calibration.CalibratedClassifierCV` class.
        clf_kwargs
            A dictionary of keyword arguments to be passed to the :py:class:`~sklearn.svm.SVC` classifier.
        """
        self.c_kernel = c_kernel
        self.calibrate = calibrate
        self.cal_method = cal_method
        clf_kwargs = clf_kwargs or {}
        self.clf = SVC(kernel=self.c_kernel, **clf_kwargs)

    def __call__(self,  c_all: np.ndarray,  bools: np.ndarray) -> np.ndarray:
        """
        The classifier's call method.

        Parameters
        ----------
        c_all
            Array containing conditioning variables for each instance.
        bools
            Boolean array marking the domain each instance belongs to (`0` for reference, `1` for test).

        Returns
        -------
        Propensity scores (the probability of being test instances).
        """
        self.clf.fit(c_all, bools)
        clf_cal = self.clf
        if self.calibrate:
            clf_cal = CalibratedClassifierCV(self.clf, method=self.cal_method, cv='prefit')
            clf_cal.fit(c_all, bools)
        prop_scores = clf_cal.predict_proba(c_all)[:, 1]
        return prop_scores
