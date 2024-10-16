---
title: Online Drift Detection on the Wine Quality Dataset
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---


In the context of deployed models, data (model queries) usually arrive sequentially and we wish to detect it as soon as possible after its occurence. One approach is to perform a test for drift every $W$ time-steps, using the $W$ samples that have arrived since the last test. Such a strategy could be implemented using any of the offline detectors implemented in `alibi-detect`, but being both sensitive to slight drift and responsive to severe drift is difficult. If the window size $W$ is too small then slight drift will be undetectable. If it is too large then the delay between test-points hampers responsiveness to severe drift.

An alternative strategy is to perform a test each time data arrives. However the usual offline methods are not applicable because the process for computing p-values is too expensive and doesn't account for correlated test outcomes when using overlapping windows of test data. 

Online detectors instead work by computing the test-statistic once using the first $W$ data points and then updating the test-statistic sequentially at low cost. When no drift has occured the test-statistic fluctuates around its expected value and once drift occurs the test-statistic starts to drift upwards. When it exceeds some preconfigured threshold value, drift is detected.

Unlike offline detectors which require the specification of a threshold p-value (a false positive rate), the online detectors in `alibi-detect` require the specification of an expected run-time (ERT) (an inverted FPR). This is the number of time-steps that we insist our detectors, on average, should run for in the absense of drift before making a false detection. Usually we would like the ERT to be large, however this results in insensitive detectors which are slow to respond when drift does occur. There is a tradeoff between the expected run time and the expected detection delay. 

To target the desired ERT, thresholds are configured during an initial configuration phase via simulation. This configuration process is only suitable when the amount reference data (most likely the training data of the model of interest) is relatively large (ideally around an order of magnitude larger than the desired ERT). Configuration can be expensive (less so with a GPU) but allows the detector to operate at low-cost during deployment. 

This notebook demonstrates online drift detection using two different two-sample distance metrics for the test-statistic, the maximum mean discrepency (MMD) and least-squared density difference (LSDD), both of which can be updated sequentially at low cost. 

### Backend

The online detectors are implemented in both the *PyTorch* and *TensorFlow* frameworks with support for CPU and GPU. Various preprocessing steps are also supported out-of-the box in Alibi Detect for both frameworks and an example will be given in this notebook. Alibi Detect does however not install PyTorch for you. Check the [PyTorch docs](https://pytorch.org/) how to do this. 

### Dataset

The [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality) consists of 4898 and 1599 samples of white and red wine respectively. Each sample has an associated quality (as determined by experts) and 11 numeric features indicating its acidity, density, pH etc. We consider the regression problem of tring to predict the quality of white wine samples given these features. We will then consider whether the model remains suitable for predicting the quality of red wine samples or whether the associated change in the underlying distribution should be considered as drift.


## Online detection with MMD and Pytorch

The Maximum Mean Discepency (MMD) is a distance-based measure between 2 distributions *p* and *q* based on the mean embeddings $\mu_{p}$ and $\mu_{q}$ in a reproducing kernel Hilbert space $F$:

$$
MMD(F, p, q) = || \mu_{p} - \mu_{q} ||^2_{F}
$$

Given reference samples $\{X_i\}_{i=1}^{N}$ and test samples $\{Y_i\}_{i=t}^{t+W}$ we may compute an unbiased estimate $\widehat{MMD}^2(F, \{X_i\}_{i=1}^N, \{Y_i\}_{i=t}^{t+W})$ of the squared MMD between the two underlying distributions. Depending on the size of the reference and test windows, $N$ and $W$ respectively, this can be relatively expensive. However, once computed it is possible to update the statistic to estimate to the squared MMD between the distributions underlying $\{X_i\}_{i=1}^{N}$ and $\{Y_i\}_{i=t+1}^{t+1+W}$ at a very low cost, making it suitable for online drift detection.

By default we use a [radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel), but users are free to pass their own kernel of preference to the detector.

```{python}
import matplotlib.pyplot as plt
import numpy as np
import torch
import tensorflow as tf
import pandas as pd
import scipy
from sklearn.decomposition import PCA

np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)
```

### Load data

First we load in the data:

```{python}
red = pd.read_csv(
    "https://storage.googleapis.com/seldon-datasets/wine_quality/winequality-red.csv", sep=';'
)
white = pd.read_csv(
    "https://storage.googleapis.com/seldon-datasets/wine_quality/winequality-white.csv", sep=';'
)
white.describe()
```

We can see that the data for both red and white wine samples take the same format.

```{python}
red.describe()
```

We shuffle and normalise the data such that each feature takes a value in \[0,1\], as does the quality we seek to predict. We assue that our model was trained on white wine samples, which therefore forms the reference distribution, and that red wine samples can be considered to be drawn from a drifted distribution.

```{python}
white, red = np.asarray(white, np.float32), np.asarray(red, np.float32)
n_white, n_red = white.shape[0], red.shape[0]

col_maxes = white.max(axis=0)
white, red = white / col_maxes, red / col_maxes
white, red = white[np.random.permutation(n_white)], red[np.random.permutation(n_red)]
X = white[:, :-1]
X_corr = red[:, :-1]
```

Although it may not be necessary on this relatively low-dimensional data for which individual features are semantically meaningful, we demonstrate how [principle component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) can be performed as a preprocessing stage to project raw data onto a lower dimensional representation which more concisely captures the factors of variation in the data. As not to bias the detector it is necessary to fit the projection using a split of the data which isn't then passed as reference data. We additionally split off some white wine samples to act as undrifted data during deployment.

```{python}
X_train = X[:(n_white//2)]
X_ref = X[(n_white//2):(3*n_white//4)]
X_h0 = X[(3*n_white//4):]
```

Now we define a PCA object to be used as a preprocessing function to project the 11-D data onto a 2-D representation. We learn the first 2 principal components on the training split of the reference data.

```{python}
pca = PCA(2)
pca.fit(X_train)
```

Hopefully the learned preprocessing step has learned a projection such that in the lower dimensional space the two samples are distinguishable.

```{python}
enc_h0 = pca.transform(X_h0)
enc_h1 = pca.transform(X_corr)

plt.scatter(enc_h0[:,0], enc_h0[:,1], alpha=0.2, color='green', label='white wine')
plt.scatter(enc_h1[:,0], enc_h1[:,1], alpha=0.2, color='red', label='red wine')
plt.legend(loc='upper right')
plt.show()
```

Now we can define our online drift detector. We specify an expected run-time (in the absence of drift) of 50 time-steps, and a window size of 10 time-steps. Upon initialising the detector thresholds will be computed using 2500 boostrap samples. These values of `ert`, `window_size` and `n_bootstraps` are lower than a typical use-case in order to demonstrate the average behaviour of the detector over a large number of runs in a reasonable time. 

```{python}
from alibi_detect.cd import MMDDriftOnline

ert = 50
window_size = 10

cd = MMDDriftOnline(
    X_ref, ert, window_size, backend='pytorch', preprocess_fn=pca.transform, n_bootstraps=2500
)
```

We now define a function which will simulate a single run and return the run-time. Note how the detector acts on single instances at a time, the run-time is considered as the time elapsed after the test-window has been filled, and that the detector is stateful and must be reset between detections.

```{python}
def time_run(cd, X, window_size):
    n = X.shape[0]
    perm = np.random.permutation(n)
    t = 0
    cd.reset_state()
    while True:
        pred = cd.predict(X[perm[t%n]])
        if pred['data']['is_drift'] == 1:
            return t
        else:
            t += 1
```

Now we look at the distribution of run-times when operating on the held-out data from the reference distribution of white wine samples. We report the average run-time, however note that the targeted run-time distribution, a Geometric distribution with mean `ert`, is very high variance so the empirical average may not be that close to `ert` over a relatively small number of runs. We can see that the detector accurately targets the desired Geometric distribution however by inspecting the linearity of a [Q-Q plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot).

```{python}
n_runs = 250
times_h0 = [time_run(cd, X_h0, window_size) for _ in range(n_runs)]
print(f"Average run-time under no-drift: {np.mean(times_h0)}")
_ = scipy.stats.probplot(np.array(times_h0), dist=scipy.stats.geom, sparams=1/ert, plot=plt)
```

If we run the detector in an identical manner but on data from the drifted distribution of red wine samples the average run-time is much lower.

```{python}
n_runs = 250
times_h1 = [time_run(cd, X_corr, window_size) for _ in range(n_runs)]
print(f"Average run-time under drift: {np.mean(times_h1)}")
```

## Online detection with LSDD and TensorFlow

Here we address the same problem but using the least squares density difference (LSDD) as the two-sample distance in a manner similar to [Bu et al. (2017)](https://ieeexplore.ieee.org/abstract/document/7890493). The LSDD between two distributions $p$ and $q$ on $\mathcal{X}$ is defined as $$LSDD(p,q) = \int_{\mathcal{X}} (p(x)-q(x))^2 \,dx$$ and also has an empirical estimate $\widehat{LSDD}(\{X_i\}_{i=1}^N, \{Y_i\}_{i=t}^{t+W})$ that can be updated at low cost as the test window is updated to $\{Y_i\}_{i=t+1}^{t+1+W}$.

We additionally show that TensorFlow can also be used as the backend and that sometimes it is not necessary to perform preprocessing, making definition of the drift detector simpler. Moreover, in the absence of a learned preprocessing stage we may use all of the reference data available.

```{python}
X_ref = np.concatenate([X_train, X_ref], axis=0)
```

And now we define the LSDD-based online drift detector, again with an `ert` of 50 and `window_size` of 10.

```{python}
from alibi_detect.cd import LSDDDriftOnline

cd = LSDDDriftOnline(
    X_ref, ert, window_size, backend='tensorflow', n_bootstraps=2500,
)
```

We run this new detector on the held out reference data and again see that in the absence of drift the distribution of run-times follows a Geometric distribution with mean `ert`.

```{python}
n_runs = 250
times_h0 = [time_run(cd, X_h0, window_size) for _ in range(n_runs)]
print(f"Average run-time under no-drift: {np.mean(times_h0)}")
_ = scipy.stats.probplot(np.array(times_h0), dist=scipy.stats.geom, sparams=1/ert, plot=plt)
```

And when drift has occured the detector is very fast to respond.

```{python}
n_runs = 250
times_h1 = [time_run(cd, X_corr, window_size) for _ in range(n_runs)]
print(f"Average run-time under drift: {np.mean(times_h1)}")
```

