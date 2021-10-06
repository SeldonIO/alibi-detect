import numpy as np
from alibi_detect.cd import FETDriftOnline

ert = 100
N = 1000
B = 10000
window_sizes = [10,20]
n_repeats = 100
p_h0 = 0.5
p_h1 = 0.3
lam = 0.99

z_ref = np.random.choice(2, N, p=[1-p_h0, p_h0])
stream_h0 = (np.random.choice(2, p=[1-p_h0, p_h0]) for _ in range(int(1e8)))
stream_h1 = (np.random.choice(2, p=[1-p_h1, p_h1]) for _ in range(int(1e8)))

dd = FETDriftOnline(z_ref, ert=ert, window_size=window_sizes, n_bootstraps=B, lam=lam)

times_h0 = []
while len(times_h0) < n_repeats:
    pred = dd.predict(next(stream_h0))
    if pred['data']['is_drift']:
        print(dd.t)
        times_h0.append(dd.t)
        dd.reset()

times_h1 = []
while len(times_h1) < n_repeats:
    pred = dd.predict(next(stream_h1))
    if pred['data']['is_drift']:
        print(dd.t)
        times_h1.append(dd.t)
        dd.reset()

art = np.array(times_h0).mean() - np.min(window_sizes) + 1
add = np.array(times_h1).mean() - np.min(window_sizes)
# Note art won't be super close to ert as we haven't averaged over initialisations.
print(f'ERT: {ert}')
print(f'ART: {art}')
print(f'ADD: {add}')
