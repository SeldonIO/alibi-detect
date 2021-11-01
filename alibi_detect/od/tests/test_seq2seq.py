from itertools import product
import numpy as np
import pytest
from alibi_detect.od import OutlierSeq2Seq
from alibi_detect.utils.perturbation import inject_outlier_ts
from alibi_detect.version import __version__

n_features = [1, 2]
seq_len = [20, 50]
threshold = [None, 5.]
threshold_perc = [90.]
return_instance_score = [True, False]
return_feature_score = [True, False]
outlier_perc = [100]
outlier_type = ['instance', 'feature']

tests = list(product(n_features, seq_len, threshold, threshold_perc,
                     return_instance_score, return_feature_score, outlier_perc, outlier_type))
n_tests = len(tests)

latent_dim = 20


@pytest.fixture
def seq2seq_params(request):
    return tests[request.param]


@pytest.mark.parametrize('seq2seq_params', list(range(n_tests)), indirect=True)
def test_seq2seq(seq2seq_params):
    # OutlierSeq2Seq parameters
    n_features, seq_len, threshold, threshold_perc, return_instance_score, \
        return_feature_score, outlier_perc, outlier_type = seq2seq_params

    # create artificial sine time series
    X = np.sin(np.linspace(-50, 50, 10000)).astype(np.float32).reshape((-1, n_features))

    # create outliers for threshold and detection
    X_threshold = inject_outlier_ts(X, perc_outlier=100 - threshold_perc, perc_window=10, n_std=10., min_std=9.).data
    X_outlier = inject_outlier_ts(X, perc_outlier=100 - threshold_perc, perc_window=10, n_std=10., min_std=9.).data

    # define architecture
    od = OutlierSeq2Seq(n_features, seq_len, threshold=threshold, latent_dim=latent_dim)

    if threshold is None:
        assert od.threshold == 0.
    else:
        assert od.threshold == threshold
    assert od.meta == {'name': 'OutlierSeq2Seq', 'detector_type': 'offline', 'data_type': 'time-series',
                       'version': __version__}

    # fit OutlierSeq2Seq
    od.fit(X, epochs=2, verbose=False)

    # create some outliers and infer threshold
    od.infer_threshold(X_threshold, threshold_perc=threshold_perc)

    # compute scores and check ranges
    fscore, iscore = od.score(X_threshold, outlier_perc=outlier_perc)
    if isinstance(od.threshold, np.ndarray):
        perc_score = 100 * (fscore < 0).astype(int).sum() / iscore.shape[0] / n_features
        assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # create outliers and make predictions
    od_preds = od.predict(X_outlier,
                          return_instance_score=return_instance_score,
                          return_feature_score=return_feature_score,
                          outlier_type=outlier_type,
                          outlier_perc=outlier_perc)
    assert od_preds['meta'] == od.meta
    if outlier_type == 'instance':
        assert od_preds['data']['is_outlier'].shape == (X.shape[0],)
        if return_instance_score:
            assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['instance_score'] > 0).astype(int).sum()
    elif outlier_type == 'feature':
        assert od_preds['data']['is_outlier'].shape == X.shape
        if return_feature_score:
            assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['feature_score'] > 0).astype(int).sum()

    if return_feature_score:
        assert od_preds['data']['feature_score'].shape == X.shape
    else:
        assert od_preds['data']['feature_score'] is None

    if return_instance_score:
        assert od_preds['data']['instance_score'].shape == (X.shape[0],)
    else:
        assert od_preds['data']['instance_score'] is None
