from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


class DomainClf(ABC):
    """
    Base class for domain classifiers used in :py:class:`~alibi_detect.cd.ContextAwareDrift`.

    The classifiers should be fit on conditioning variables `c_all` and their domain `bools` as input (`0` for ref, `1`
    test). They should predict propensity scores (probability of being test instances) as output.
    Classifiers should possess a calibrate method to calibrate the propensity scores.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs: dict):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, c_all: np.ndarray, bools: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def calibrate(self, c_all: np.ndarray, bools: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, c_all: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class SVCDomainClf(DomainClf):
    def __init__(self,
                 c_kernel: Callable,
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
        cal_method
            The method to be used to calibrate the detector. This should be a method accepted by the scikit-learn
            :py:class:`~sklearn.calibration.CalibratedClassifierCV` class.
        clf_kwargs
            A dictionary of keyword arguments to be passed to the :py:class:`~sklearn.svm.SVC` classifier.
        """
        self.c_kernel = c_kernel
        self.cal_method = cal_method
        clf_kwargs = clf_kwargs or {}
        self.clf = SVC(kernel=self.c_kernel, **clf_kwargs)

    def fit(self, c_all: np.ndarray, bools: np.ndarray):
        """
        Method to fit the classifier.

        Parameters
        ----------
        c_all
            Array containing conditioning variables for each instance.
        bools
            Boolean array marking the domain each instance belongs to (`0` for reference, `1` for test).
        """
        clf = self.clf
        clf.fit(c_all, bools)
        self.clf = clf

    def calibrate(self, c_all: np.ndarray, bools: np.ndarray):
        """
        Method to calibrate the classifier's predicted probabilities.

        Parameters
        ----------
        c_all
            Array containing conditioning variables for each instance.
        bools
            Boolean array marking the domain each instance belongs to (`0` for reference, `1` for test).
        """
        clf = CalibratedClassifierCV(self.clf, method=self.cal_method, cv='prefit')
        clf.fit(c_all, bools)
        self.clf = clf

    def predict(self,  c_all: np.ndarray) -> np.ndarray:
        """
        The classifier's predict method.

        Parameters
        ----------
        c_all
            Array containing conditioning variables for each instance.

        Returns
        -------
        Propensity scores (the probability of being test instances).
        """
        return self.clf.predict_proba(c_all)[:, 1]
