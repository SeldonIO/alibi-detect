import datetime
import fbprophet
from itertools import product
import numpy as np
import pandas as pd
import pytest
from alibi_detect.od import OutlierProphet
from alibi_detect.version import __version__

growth = ['linear', 'logistic']
return_instance_score = [True, False]
return_forecast = [True, False]

d_fit = {
    'ds': pd.date_range(pd.datetime.today(), periods=100),
    'y': np.random.randn(100)
}
df_fit = pd.DataFrame(data=d_fit)

d_test = {
    'ds': pd.date_range(d_fit['ds'][-1] + datetime.timedelta(1), periods=100),
    'y': np.random.randn(100)
}
df_test = pd.DataFrame(data=d_test)

tests = list(product(growth, return_instance_score, return_forecast))
n_tests = len(tests)


@pytest.fixture
def prophet_params(request):
    return tests[request.param]


@pytest.mark.parametrize('prophet_params', list(range(n_tests)), indirect=True)
def test_prophet(prophet_params):
    growth, return_instance_score, return_forecast = prophet_params
    od = OutlierProphet(growth=growth)
    assert isinstance(od.model, fbprophet.forecaster.Prophet)
    assert od.meta == {'name': 'OutlierProphet', 'detector_type': 'offline', 'data_type': 'time-series',
                       'version': __version__}
    if growth == 'logistic':
        df_fit['cap'] = 10.
        df_test['cap'] = 10.
    od.fit(df_fit)
    forecast = od.score(df_test)
    fcst_cols = list(forecast.columns)
    check_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'score', 'y']
    assert all(check_col in fcst_cols for check_col in check_cols)
    od_preds = od.predict(df_test,
                          return_instance_score=return_instance_score,
                          return_forecast=return_forecast)
    assert od_preds['meta'] == od.meta
    assert (od_preds['data']['is_outlier']['ds'] == df_test['ds']).all()
    assert od_preds['data']['is_outlier']['is_outlier'].shape[0] == df_test['ds'].shape[0]
    if not return_instance_score:
        assert od_preds['data']['instance_score'] is None
    if not return_forecast:
        with pytest.raises(KeyError):
            od_preds['data']['forecast']
