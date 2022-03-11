from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


class _DomainClf(ABC):
    """
    Base class for domain classifiers used in :py:class:`~alibi_detect.cd.ContextAwareDrift`.The `SVCDomainClf` is
    currently hardcoded into the detector. Therefore, for now, these classes (and the domain_clf submodule) are
    kept private. This is subject to change in the future.

    The classifiers should be fit on conditioning variables `x` and their domain `y` (`0` for ref, `1`
    test). They should predict propensity scores (probability of being test instances) as output.
    Classifiers should possess a calibrate method to calibrate the propensity scores.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs: dict):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def calibrate(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class _SVCDomainClf(_DomainClf):
    def __init__(self,
                 kernel: Callable,
                 cal_method: str = 'sigmoid',
                 clf_kwargs: dict = None):
        """
        A domain classifier using the scikit-learn Support Vector Classifier
        (:py:class:`~sklearn.svm.SVC`). An SVC is fitted on all the
        data, with scores (that optimise hinge loss) mapped onto probabilities using logistic regression.

        Parameters
        ----------
        kernel
            Kernel used to pre-compute the kernel matrix from data matrices.
        cal_method
            The method to be used to calibrate the detector. This should be a method accepted by the scikit-learn
            :py:class:`~sklearn.calibration.CalibratedClassifierCV` class.
        clf_kwargs
            A dictionary of keyword arguments to be passed to the :py:class:`~sklearn.svm.SVC` classifier.
        """
        self.kernel = kernel
        self.cal_method = cal_method
        clf_kwargs = clf_kwargs or {}
        self.clf = SVC(kernel=self.kernel, **clf_kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Method to fit the classifier.

        Parameters
        ----------
        x
            Array containing conditioning variables for each instance.
        y
            Boolean array marking the domain each instance belongs to (`0` for reference, `1` for test).
        """
        clf = self.clf
        clf.fit(x, y)
        self.clf = clf

    def calibrate(self, x: np.ndarray, y: np.ndarray):
        """
        Method to calibrate the classifier's predicted probabilities.

        Parameters
        ----------
        x
            Array containing conditioning variables for each instance.
        y
            Boolean array marking the domain each instance belongs to (`0` for reference, `1` for test).
        """
        clf = CalibratedClassifierCV(self.clf, method=self.cal_method, cv='prefit')
        clf.fit(x, y)
        self.clf = clf

    def predict(self,  x: np.ndarray) -> np.ndarray:
        """
        The classifier's predict method.

        Parameters
        ----------
        x
            Array containing conditioning variables for each instance.

        Returns
        -------
        Propensity scores (the probability of being test instances).
        """
        return self.clf.predict_proba(x)[:, 1]
