import numpy as np
import matplotlib.pyplot as plt

N = 400
data = np.load('cvm_benchmarks_N%d.npz' %N)
times = data['times']
Brange = data['Brange']
devices = data['devices'][::-1]

fig, ax = plt.subplots()
ax.set_xlabel(r'$B$')
ax.set_ylabel('Wall time (s)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True)
for d, device in enumerate(devices):
    ax.errorbar(Brange, times[:,d,:].mean(axis=-1), yerr=1.96*times[:,d,:].std(axis=-1), marker='o', ms=4, capsize=6, label=device)
ax.legend()


# Speedup
fig, ax = plt.subplots()
ax.set_xlabel(r'$B$')
ax.set_ylabel('Speed-up')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.grid(True)
for d, device in enumerate(devices[1:]):
    speedup = times[:,d+1,:]/times[:,0,:]
    ax.errorbar(Brange, speedup.mean(axis=-1), yerr=1.96*speedup.std(axis=-1), marker='o', ms=4, capsize=6, label=device)


plt.show()
