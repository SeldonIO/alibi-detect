import logging
import numpy as np
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union
from sklearn.base import clone, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from alibi_detect.cd.base import BaseClassifierDrift

logger = logging.getLogger(__name__)


class ClassifierDriftSklearn(BaseClassifierDrift):
    def __init__(
            self,
            x_ref: np.ndarray,
            model: ClassifierMixin,
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            preds_type: str = 'probs',
            binarize_preds: bool = False,
            train_size: Optional[float] = .75,
            n_folds: Optional[int] = None,
            retrain_from_scratch: bool = True,
            seed: int = 0,
            use_calibration: bool = False,
            calibration_kwargs: Optional[dict] = None,
            data_type: Optional[str] = None,
    ) -> None:
        """
        Classifier-based drift detector. The classifier is trained on a fraction of the combined
        reference and test data and drift is detected on the remaining data. To use all the data
        to detect drift, a stratified cross-validation scheme can be chosen.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        model
            Sklearn classification model used for drift detection.
        p_val
            p-value used for the significance of the test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        preds_type
            Whether the model outputs 'probs' or 'scores'.
        binarize_preds
            Whether to test for discrepancy on soft (e.g. probs/scores) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the classifier.
            The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`.
        n_folds
            Optional number of stratified folds used for training. The model preds are then calculated
            on all the out-of-fold predictions. This allows to leverage all the reference and test data
            for drift detection at the expense of longer computation. If both `train_size` and `n_folds`
            are specified, `n_folds` is prioritized.
        retrain_from_scratch
            Whether the classifier should be retrained from scratch for each set of test data or whether
            it should instead continue training from where it left off on the previous set.
        seed
            Optional random seed for fold selection.
        use_calibration
            Whether to use calibration. Whether to use calibration. Calibration can be used on top of any model.
        calibration_kwargs
            Optional additional kwargs for calibration.
            See https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
            for more details.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            preds_type=preds_type,
            binarize_preds=binarize_preds,
            train_size=train_size,
            n_folds=n_folds,
            retrain_from_scratch=retrain_from_scratch,
            seed=seed,
            data_type=data_type
        )

        if preds_type not in ['probs', 'scores']:
            raise ValueError("'preds_type' should be 'probs' or 'scores'")

        self.meta.update({'backend': 'sklearn'})
        self.original_model = model
        self.use_calibration = use_calibration
        self.calibration_kwargs = dict() if calibration_kwargs is None else calibration_kwargs
        self.model = self._clone_model()  # type: ClassifierMixin

    def _has_predict_proba(self, model) -> bool:
        try:
            # taking self.x_ref[0].shape to overcome bot cases when self.x_ref is np.ndarray or list
            model.predict_proba(np.zeros((1, ) + self.x_ref[0].shape))
            has_predict_proba = True
        except NotFittedError:
            has_predict_proba = True
        except AttributeError:
            has_predict_proba = False
        return has_predict_proba

    def _clone_model(self):
        model = clone(self.original_model)

        # equivalence between `retrain_from_scratch` and `warm_start`
        if not self.retrain_from_scratch:
            if hasattr(model, 'warm_start'):
                model.warm_start = True
                logger.warning('`retrain_from_scratch=False` sets automatically the parameter `warm_start=True` '
                               'for the given classifier. Please consult the documentation to ensure that the '
                               '`warm_start=True` is applicable in the current context (i.e., for tree-based '
                               'models such as RandomForest, setting `warm_start=True` is not applicable since the '
                               'fit function expects the same dataset and an update/increase in the number of '
                               'estimators - previous fitted estimators will be kept frozen while the new ones '
                               'will be fitted).')
            else:
                logger.warning('Current classifier does not support `warm_start`. The model will be retrained '
                               'from scratch every iteration.')
        else:
            if hasattr(model, 'warm_start'):
                model.warm_start = False
                logger.warning('`retrain_from_scratch=True` sets automatically the parameter `warm_start=False`.')

        if self.preds_type == 'probs':
            # calibrate the model if user specified.
            if self.use_calibration:
                model = CalibratedClassifierCV(base_estimator=model, **self.calibration_kwargs)
                logger.warning('Using calibration to obtain the prediction probabilities.')

            # check if it has predict proba. Cannot be checked via `hasattr` due to the same issue in SVC (see below)
            has_predict_proba = self._has_predict_proba(model)

            # if the binarize_preds=True, we don't really need the probabilities as in test_probs will be rounded
            # to the closest integer (i.e., to 0 or 1) according to the predicted probability. Thus, we can define
            # a hard label predict_proba based on the predict method
            if self.binarize_preds and (not has_predict_proba):
                if not hasattr(model, 'predict'):
                    raise AttributeError('Trying to use a model which does not support `predict`.')

                def predict_proba(self, X):
                    return np.eye(2)[self.predict(X).astype(np.int32)]

                # add predict_proba method. Overwriting predict_proba is not possible for SVC due
                # to @available_if(_check_proba)
                # Check link: https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/svm/_base.py#L807
                setattr(model, 'aux_predict_proba', partial(predict_proba, model))
            elif has_predict_proba:
                setattr(model, 'aux_predict_proba', model.predict_proba)

            # at this point the model does not have any predict_proba, thus the test can not be performed.
            if not hasattr(model, 'aux_predict_proba'):
                raise AttributeError("Trying to use a model which does not support `predict_proba` with "
                                     "`preds_type='probs'`. Set (`use_calibration=True`, `calibration_kwargs`) or "
                                     "(`binarize_preds=True`).")

        else:
            if self.use_calibration:
                logger.warning("No calibration is performed when `preds_type='scores'`.")

            if self.binarize_preds:
                raise ValueError("`binarize_preds` must be `False` when `preds_type='scores'`.")

            if not hasattr(model, 'decision_function'):
                raise AttributeError("Trying to use a model which does not support `decision_function` with "
                                     "`preds_type='scores'`.")

            # need to put the scores in the format expected by test function, which requires to duplicate the
            # scores along axis=1
            def predict_proba(self, X):
                scores = self.decision_function(X).reshape(-1, 1)
                return np.tile(scores, reps=2)

            # add predict_proba method
            setattr(model, 'aux_predict_proba', partial(predict_proba, model))

        return model

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Compute the out-of-fold drift metric such as the accuracy from a classifier
        trained to distinguish the reference data from the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value, a notion of distance between the trained classifier's out-of-fold performance \
        and that which we'd expect under the null assumption of no drift, \
        and the out-of-fold classifier model prediction probabilities on the reference and test data

        """
        x_ref, x = self.preprocess(x)
        n_ref, n_cur = len(x_ref), len(x)
        x, y, splits = self.get_splits(x_ref, x)

        # iterate over folds: train a new model for each fold and make out-of-fold (oof) predictions
        probs_oof_list, idx_oof_list = [], []
        for idx_tr, idx_te in splits:
            y_tr = y[idx_tr]

            if isinstance(x, np.ndarray):
                x_tr, x_te = x[idx_tr], x[idx_te]
            elif isinstance(x, list):
                x_tr, x_te = [x[_] for _ in idx_tr], [x[_] for _ in idx_te]
            else:
                raise TypeError(f'x needs to be of type np.ndarray or list and not {type(x)}.')

            self.model.fit(x_tr, y_tr)
            probs = self.model.aux_predict_proba(x_te)
            probs_oof_list.append(probs)
            idx_oof_list.append(idx_te)

        probs_oof = np.concatenate(probs_oof_list, axis=0)
        idx_oof = np.concatenate(idx_oof_list, axis=0)
        y_oof = y[idx_oof]
        p_val, dist = self.test_probs(y_oof, probs_oof, n_ref, n_cur)
        probs_sort = probs_oof[np.argsort(idx_oof)]
        return p_val, dist, probs_sort[:n_ref, 1], probs_sort[n_ref:, 1]
