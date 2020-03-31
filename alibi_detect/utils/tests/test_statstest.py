from itertools import product
import numpy as np
import pytest
from alibi_detect.utils.statstest import fdr

q_val = [.05, .1, .25]
n_p = 1000
p_vals = [
    {'is_below': True, 'p_val': np.zeros(n_p)},
    {'is_below': False, 'p_val': np.zeros(n_p)}
]
tests_fdr = list(product(q_val, p_vals))
n_tests_fdr = len(tests_fdr)


@pytest.fixture
def fdr_params(request):
    return tests_fdr[request.param]


@pytest.mark.parametrize('fdr_params', list(range(n_tests_fdr)), indirect=True)
def test_fdr(fdr_params):
    q_val, p_vals = fdr_params
    if p_vals['is_below'] and p_vals['p_val'].max() == 0:
        p_val = p_vals['p_val'] + q_val - 1e-5
    elif not p_vals['is_below'] and p_vals['p_val'].max() == 0:
        p_val = p_vals['p_val'] + q_val
    else:
        p_val = p_vals['p_val'].copy()
    below_threshold = fdr(p_val, q_val)
    assert below_threshold == p_vals['is_below']
