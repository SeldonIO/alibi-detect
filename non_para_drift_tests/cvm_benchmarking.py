import numpy as np
import time
from alibi_detect.cd import CVMDriftOnline

nfolds = 3
#Brange = np.logspace(4,5.0,6, dtype=int) # Number of bootstrap simulations for threshold estimation.
Brange = [50000]
devices = ['parallel','cpu']

ert = 100  # Desired expected runtime
N = 200 # Size of reference set
window_sizes = [10,20]
x_ref = np.random.uniform(0, 1, N)

times = np.zeros([len(Brange), len(devices), nfolds])
Brange = Brange[::-1] # start with largest
for b, B in enumerate(Brange):
    for d, device in enumerate(devices):
        print('\nn_bootstraps %d, Device = %s' % (B,device))
        for t in range(nfolds):
            t0 = time.time()
            dd = CVMDriftOnline(x_ref, ert=ert, window_size=window_sizes, n_bootstraps=B,
                                verbose=True, device=device)
            times[b,d,t] = time.time() - t0
        print('Wall time = %.3fs +- %.2gs' % (times[b,d,:].mean(),times[b,d,:].std()))

np.savez('cvm_benchmarks_N%d.npz' % N, times=times, Brange=Brange, devices=devices)


