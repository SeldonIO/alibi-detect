from fbprophet import Prophet

# incl. saturating growth/min
# tune nb of changepoints + `changepoint_range` (default=80% of dataset) + `changepoint_prior_scale`
# higher `changepoint_prior_scale`? -> more predicted uncertainty b/c more room for overfitting
# add custom + country holiday fn
# allow for nb fourier series to model seasonality to change (default=10)
# default = weekly / yearly -> allow custom! E.g. monthly / quarterly / hourly
# conditional seasonalities!
# add holiday / seasonality regularization via `holidays_prior_scale` / `seasonality_prior_scale`
# allow for multiplicative seasonality via `seasonality_mode`=`multiplicative`
# use uncertainty observations combined/separately: uncertainty in trend / seasonality / observation noise
# make sure the width of the uncertainty intervals (default=80%) is set OK via `interval_width`
# for uncertainty in seasonality -> `mcmc_samples`>0! But very costly! (minutes vs. seconds)
# important that there are no outliers in the data to fit -> can even set them to NA
# make sure timestamp format is correct
# add plotting fn

# threshold in nb stdev or something similar?

# general things for time series:
# - predict a window ahead -> if the predictions over that window are above a threshold: outlier!

from fbprophet import Prophet
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, outlier_prediction_dict

logger = logging.getLogger(__name__)


class OutlierProphet(BaseDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 window: int = 1,
                 ) -> None:
        """
        Outlier detector for time series data using fbprophet.

        Parameters
        ----------
        threshold
            Threshold used for outlier score to determine outliers.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.window = window

    def fit(self,
            df: pd.DataFrame = None,
            growth: str = 'linear',
            holidays: pd.DataFrame = None,
            holidays_prior_scale: float = 10.,  # higher = more flexible trend so more overfitting
            changepoint_range: float = .8,  # higher = more changepoints so more overfitting
            changepoint_prior_scale: float = .05,  # higher = more flexible trend so more overfitting
            seasonality_mode: str = 'additive',  # additive or multiplicative
            daily_seasonality: Union[str, int] = 'auto',
            weekly_seasonality: Union[str, int] = 'auto',
            yearly_seasonality: Union[str, int] = 'auto',
            add_seasonality: List[Dict[str, float, int, float, str]] = None,  # name, period, fourier_order, prior_scale, mode
            seasonality_prior_scale: float = 10.,  # higher = more flexible trend so more overfitting
            uncertainty_samples: int = 1000,
            country_holidays: str = None,  # e.g. 'US' to add US holidays; https://github.com/dr-prodigy/python-holidays
            ) -> None:
        # TODO: add conditional seasonalities
        kwargs = {  # TODO: add remaining kwargs
            'growth': growth,
            'holidays': holidays,
            'changepoint_range': changepoint_range,
            'changepoint_prior_scale': changepoint_prior_scale
        }
        self.model = Prophet(**kwargs)
        if country_holidays:
            self.model.add_country_holidays(country_name=country_holidays)
        for s in add_seasonality:
            self.model.add_seasonality(**s)
        self.model.fit(df)

    def infer_threshold(self, X: np.ndarray) -> None:
        pass

    def score(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict(self, X: np.ndarray) -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        pass