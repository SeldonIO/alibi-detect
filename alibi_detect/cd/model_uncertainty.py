import numpy as np
from typing import Callable, Dict, Optional, Union
from functools import partial
from alibi_detect.cd.ks import KSDrift
from alibi_detect.cd.chisquare import ChiSquareDrift
from alibi_detect.cd.preprocess import classifier_uncertainty, regressor_uncertainty


class ClassifierUncertaintyDrift:
    def __init__(
            self,
            x_ref: np.ndarray,
            model: Callable,
            p_val: float = .05,
            backend: Optional[str] = None,
            update_x_ref: Optional[Dict[str, int]] = None,
            preds_type: str = 'probs',
            uncertainty_type: str = 'entropy',
            margin_width: float = 0.1,
            batch_size: int = 32,
            device: Optional[str] = None,
            tokenizer: Optional[Callable] = None,
            max_len: Optional[int] = None,
            data_type: Optional[str] = None,
    ) -> None:
        """
        Test for a change in the number of instances falling into regions on which the model is uncertain.
        Performs either a K-S test on prediction entropies or Chi-squared test on 0-1 indicators of predictions
        falling into a margin of uncertainty (e.g. probs falling into [0.45, 0.55] in binary case).

        Parameters
        ----------
        x_ref
            Data used as reference distribution. Should be disjoint from the data the model was trained on
            for accurate p-values.
        model
            Classification model outputting class probabilities (or logits)
        backend
            Backend to use if model requires batch prediction. Options are 'tensorflow' or 'pytorch'.
        p_val
            p-value used for the significance of the test.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preds_type
            Type of prediction output by the model. Options are 'probs' (in [0,1]) or 'logits' (in [-inf,inf]).
        uncertainty_type
            Method for determining the model's uncertainty for a given instance. Options are 'entropy' or 'margin'.
        margin_width
            Width of the margin if uncertainty_type = 'margin'. The model is considered uncertain on an instance
            if the highest two class probabilities it assigns to the instance differ by less than margin_width.
        batch_size
            Batch size used to evaluate model. Only relavent when backend has been specified for batch prediction.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
        tokenizer
            Optional tokenizer for NLP models.
        max_len
            Optional max token length for NLP models.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """

        preprocess_fn = partial(
            classifier_uncertainty,
            model=model,
            backend=backend,
            preds_type=preds_type,
            uncertainty_type=uncertainty_type,
            margin_width=margin_width,
            batch_size=batch_size,
            device=device,
            tokenizer=tokenizer,
            max_len=max_len
        )

        self._detector: Union[KSDrift, ChiSquareDrift]

        if uncertainty_type == 'entropy':
            self._detector = KSDrift(
                x_ref=x_ref,
                p_val=p_val,
                preprocess_x_ref=True,
                update_x_ref=update_x_ref,
                preprocess_fn=preprocess_fn,
                data_type=data_type
            )
        elif uncertainty_type == 'margin':
            self._detector = ChiSquareDrift(
                x_ref=x_ref,
                p_val=p_val,
                preprocess_x_ref=True,
                update_x_ref=update_x_ref,
                preprocess_fn=preprocess_fn,
                data_type=data_type
            )
        else:
            raise NotImplementedError("Only uncertainty types 'entropy' or 'margin' supported.")

        self.meta = self._detector.meta
        self.meta['name'] = 'ClassifierUncertaintyDrift'
        self.meta['preds_type'] = preds_type
        self.meta['uncertainty_type'] = uncertainty_type
        if uncertainty_type == 'margin':
            self.meta['margin_width'] = margin_width

    def predict(self, x: np.ndarray,  return_p_val: bool = True,
                return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_p_val
            Whether to return the p-value of the test.
        return_distance
            Whether to return the corresponding test statistic (K-S for 'entropy', Chi2 for 'margin').
        # TODO: Offer to return difference in uncertainty

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold and test statistic.
        """
        return self._detector.predict(x, return_p_val=return_p_val, return_distance=return_distance)


class RegressorUncertaintyDrift:
    def __init__(
            self,
            x_ref: np.ndarray,
            model: Callable,
            p_val: float = .05,
            backend: Optional[str] = None,
            update_x_ref: Optional[Dict[str, int]] = None,
            uncertainty_type: str = 'dropout',
            n_evals: int = 25,
            batch_size: int = 32,
            device: Optional[str] = None,
            tokenizer: Optional[Callable] = None,
            max_len: Optional[int] = None,
            data_type: Optional[str] = None,
    ) -> None:
        """
        Test for a change in the number of instances falling into regions on which the model is uncertain.
        Performs either a K-S test on uncertainties estimated from an preditive ensemble given either
        explicitly or implicitly as a model with dropout layers.

        Parameters
        ----------
        x_ref
            Data used as reference distribution. Should be disjoint from the data the model was trained on
            for accurate p-values.
        model
            Classification model outputting class probabilities (or logits)
        backend
            Backend to use if model requires batch prediction. Options are 'tensorflow' or 'pytorch'.
        p_val
            p-value used for the significance of the test.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        uncertainty_type
            Method for determining the model's uncertainty for a given instance. Options are 'mc_dropout' or 'ensemble'.
            The former should output a scalar per instance. The latter should output a vector of predictions
            per instance.
        n_evals:
            The number of times to evaluate the model under different dropout configurations. Only relavent when using
            the 'mc_dropout' uncertainty type.
        batch_size
            Batch size used to evaluate model. Only relavent when backend has been specified for batch prediction.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
        tokenizer
            Optional tokenizer for NLP models.
        max_len
            Optional max token length for NLP models.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """

        preprocess_fn = partial(
            regressor_uncertainty,
            model=model,
            backend=backend,
            uncertainty_type=uncertainty_type,
            batch_size=batch_size,
            device=device,
            tokenizer=tokenizer,
            max_len=max_len
        )

        self._detector = KSDrift(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_x_ref=True,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            data_type=data_type
        )

        self.meta = self._detector.meta
        self.meta['name'] = 'RegressorUncertaintyDrift'
        self.meta['uncertainty_type'] = uncertainty_type
        if uncertainty_type == 'mc_dropout':
            self.meta['n_evals'] = n_evals

    def predict(self, x: np.ndarray,  return_p_val: bool = True,
                return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_p_val
            Whether to return the p-value of the test.
        return_distance
            Whether to return the K-S test statistic
        # TODO: Offer to return difference in uncertainty

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold and test statistic.
        """
        return self._detector.predict(x, return_p_val=return_p_val, return_distance=return_distance)
