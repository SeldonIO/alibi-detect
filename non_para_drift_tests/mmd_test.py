import numpy as np
from alibi_detect.cd import MMDDriftOnline

ert = 150  # Desired expected runtime
N = 100  # Size of reference set
B = 5000  # Number of bootstrap simulations for threshold estimation. 100k more realistic?
window_size = 10

# Note how these are diff to the Gaussian used for configuration.
x_ref = np.random.uniform(0, 1, N)
stream_h0 = iter(lambda: np.random.uniform(), 1)
stream_h1 = iter(lambda: np.random.exponential(), 1)

dd = MMDDriftOnline(x_ref, ert=ert, window_size=window_size, n_bootstraps=B)

n = 0
while n < 20:
    a = np.array([np.random.uniform()])
    # a = np.array([np.random.randn()])
    print(dd.predict(a)['data']['is_drift'])
    n += 1
