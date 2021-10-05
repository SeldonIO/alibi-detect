import numpy as np
from alibi_detect.cd import CVMDriftOnline

ert = 150  # Desired expected runtime
N = 100 # Size of reference set
B = 5000 # Number of bootstrap simulations for threshold estimation. 100k more realistic?
window_sizes = [10,20]

# Note how these are diff to the Gaussian used for configuration.
x_ref = np.random.uniform(0, 1, N)
stream_h0 = iter(lambda: np.random.uniform(), 1)
stream_h1 = iter(lambda: np.random.exponential(), 1)

dd = CVMDriftOnline(x_ref, ert=ert, window_size=window_sizes, n_bootstraps=B, verbose=True, device='parallel')

n = 0
while n < 20:
    a = np.array([np.random.uniform()])
    #a = np.array([np.random.randn()])
    print(dd.predict(a)['data']['is_drift'])
    n += 1
