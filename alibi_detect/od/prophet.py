try:
    from fbprophet import Prophet

    PROPHET_INSTALLED = True
except ImportError:
    PROPHET_INSTALLED = False

import logging
import pandas as pd
from typing import Dict, List, Union
from alibi_detect.base import BaseDetector, FitMixin, outlier_prediction_dict

logger = logging.getLogger(__name__)


class OutlierProphet(BaseDetector, FitMixin):

    def __init__(self,
                 threshold: float = .8,
                 growth: str = 'linear',
                 cap: float = None,
                 holidays: pd.DataFrame = None,
                 holidays_prior_scale: float = 10.,
                 country_holidays: str = None,
                 changepoint_prior_scale: float = .05,
                 changepoint_range: float = .8,
                 seasonality_mode: str = 'additive',
                 daily_seasonality: Union[str, bool, int] = 'auto',
                 weekly_seasonality: Union[str, bool, int] = 'auto',
                 yearly_seasonality: Union[str, bool, int] = 'auto',
                 add_seasonality: List = None,
                 seasonality_prior_scale: float = 10.,
                 uncertainty_samples: int = 1000,
                 mcmc_samples: int = 0
                 ) -> None:
        """
        Outlier detector for time series data using fbprophet.
        See https://facebook.github.io/prophet/ for more details.

        Parameters
        ----------
        threshold
            Width of the uncertainty intervals of the forecast, used as outlier threshold.
            Equivalent to `interval_width`. If the instance lies outside of the uncertainty intervals,
            it is flagged as an outlier. If `mcmc_samples` equals 0, it is the uncertainty in the trend
            using the MAP estimate of the extrapolated model. If `mcmc_samples` >0, then uncertainty
            over all parameters is used.
        growth
            'linear' or 'logistic' to specify a linear or logistic trend.
        cap
            Growth cap in case growth equals 'logistic'.
        holidays
            pandas DataFrame with columns `holiday` (string) and `ds` (dates) and optionally
            columns `lower_window` and `upper_window` which specify a range of days around
            the date to be included as holidays.
        holidays_prior_scale
            Parameter controlling the strength of the holiday components model.
            Higher values imply a more flexible trend, more prone to more overfitting.
        country_holidays
            Include country-specific holidays via country abbreviations.
            The holidays for each country are provided by the holidays package in Python.
            A list of available countries and the country name to use is available on:
            https://github.com/dr-prodigy/python-holidays. Additionally, Prophet includes holidays for:
            Brazil (BR), Indonesia (ID), India (IN), Malaysia (MY), Vietnam (VN), Thailand (TH),
            Philippines (PH), Turkey (TU), Pakistan (PK), Bangladesh (BD), Egypt (EG), China (CN) and Russian (RU).
        changepoint_prior_scale
            Parameter controlling the flexibility of the automatic changepoint selection.
            Large values will allow many changepoints, potentially leading to overfitting.
        changepoint_range
            Proportion of history in which trend changepoints will be estimated.
            Higher values means more changepoints, potentially leading to overfitting.
        seasonality_mode
            Either 'additive' or 'multiplicative'.
        daily_seasonality
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        weekly_seasonality
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        yearly_seasonality
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        add_seasonality
            Manually add one or more seasonality components. Pass a list of dicts containing the keys
            `name`, `period`, `fourier_order` (obligatory), `prior_scale` and `mode` (optional).
        seasonality_prior_scale
            Parameter controlling the strength of the seasonality model. Larger values allow the model to
            fit larger seasonal fluctuations, potentially leading to overfitting.
        uncertainty_samples
            Number of simulated draws used to estimate uncertainty intervals.
        mcmc_samples
            If >0, will do full Bayesian inference with the specified number of MCMC samples.
            If 0, will do MAP estimation.
        """
        super().__init__()

        # initialize Prophet model
        # TODO: add conditional seasonalities
        kwargs = {
            'growth': growth,
            'interval_width': threshold,
            'holidays': holidays,
            'holidays_prior_scale': holidays_prior_scale,
            'changepoint_prior_scale': changepoint_prior_scale,
            'changepoint_range': changepoint_range,
            'seasonality_mode': seasonality_mode,
            'daily_seasonality': daily_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'yearly_seasonality': yearly_seasonality,
            'seasonality_prior_scale': seasonality_prior_scale,
            'uncertainty_samples': uncertainty_samples,
            'mcmc_samples': mcmc_samples
        }
        self.model = Prophet(**kwargs)
        if country_holidays:
            self.model.add_country_holidays(country_name=country_holidays)
        if add_seasonality:
            for s in add_seasonality:
                self.model.add_seasonality(**s)
        self.cap = cap

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = 'time-series'

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit Prophet model on normal (inlier) data.

        Parameters
        ----------
        df
            Dataframe with columns `ds` with timestamps and `y` with target values.
        """
        if self.cap:
            df['cap'] = self.cap
        self.model.fit(df)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute outlier scores.

        Parameters
        ----------
        df
            DataFrame with columns `ds` with timestamps and `y` with values which
            need to be flagged as outlier or not.

        Returns
        -------
        Array with outlier scores for each instance in the batch.
        """
        if self.cap:
            df['cap'] = self.cap
        forecast = self.model.predict(df)
        forecast['y'] = df['y'].values
        forecast['score'] = (
                (forecast['y'] - forecast['yhat_upper']) * (forecast['y'] >= forecast['yhat']) +
                (forecast['yhat_lower'] - forecast['y']) * (forecast['y'] < forecast['yhat'])
        )
        return forecast

    def predict(self,
                df: pd.DataFrame,
                return_instance_score: bool = True,
                return_forecast: bool = True
                ) -> Dict[Dict[str, str], Dict[pd.DataFrame, pd.DataFrame]]:
        """
        Compute outlier scores and transform into outlier predictions.

        Parameters
        ----------
        df
            DataFrame with columns `ds` with timestamps and `y` with values which
            need to be flagged as outlier or not.
        return_instance_score
            Whether to return instance level outlier scores.
        return_forecast
            Whether to return the model forecast.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the outlier predictions, instance level outlier scores and the model forecast.
        """
        # compute outlier scores
        forecast = self.score(df)
        iscore = pd.DataFrame(data={
            'ds': df['ds'].values,
            'instance_score': forecast['score']
        })

        # values above threshold are outliers
        outlier_pred = pd.DataFrame(data={
            'ds': df['ds'].values,
            'is_outlier': (forecast['score'] > 0.).astype(int)
        })

        # populate output dict
        od = outlier_prediction_dict()
        od['meta'] = self.meta
        od['data']['is_outlier'] = outlier_pred
        if return_instance_score:
            od['data']['instance_score'] = iscore
        if return_forecast:
            od['data']['forecast'] = forecast
        return od
