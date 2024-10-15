---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../api/alibi_detect.datasets.rst)

## Overview

The package also contains functionality in `alibi_detect.datasets` to easily fetch a number of datasets for different modalities. For each dataset either the data and labels or a *Bunch* object with the data, labels and optional metadata are returned. Example:

```python
from alibi_detect.datasets import fetch_ecg

(X_train, y_train), (X_test, y_test) = fetch_ecg(return_X_y=True)
```

### Sequential Data and Time Series

**Genome Dataset**: `fetch_genome`

  - Bacteria genomics dataset for out-of-distribution detection, released as part of [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845). From the original *TL;DR*: *The dataset contains genomic sequences of 250 base pairs from 10 in-distribution bacteria classes for training, 60 OOD bacteria classes for validation, and another 60 different OOD bacteria classes for test*. There are respectively 1, 7 and again 7 million sequences in the training, validation and test sets. For detailed info on the dataset check the [README](https://storage.cloud.google.com/seldon-datasets/genome/readme.docx?organizationId=156002945562).
  
  
```python
from alibi_detect.datasets import fetch_genome

(X_train, y_train), (X_val, y_val), (X_test, y_test) = fetch_genome(return_X_y=True)
```


**ECG 5000**: `fetch_ecg`

  - 5000 ECG's, originally obtained from [Physionet](https://archive.physionet.org/cgi-bin/atm/ATM).


**NAB**: `fetch_nab`

  - Any univariate time series in a DataFrame from the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB). A list with the available time series can be retrieved using `alibi_detect.datasets.get_list_nab()`.


### Images

**CIFAR-10-C**: `fetch_cifar10c`

  - CIFAR-10-C ([Hendrycks & Dietterich, 2019](https://arxiv.org/abs/1903.12261)) contains the test set of CIFAR-10, but corrupted and perturbed by various types of noise, blur, brightness etc. at different levels of severity, leading to a gradual decline in a classification model's performance trained on CIFAR-10. `fetch_cifar10c` allows you to pick any severity level or corruption type. The list with available corruption types can be retrieved with `alibi_detect.datasets.corruption_types_cifar10c()`. The dataset can be used in research on robustness and drift. The original data can be found [here](https://zenodo.org/record/2535967#.XnAM2nX7RNw). Example:
  
  
```python
from alibi_detect.datasets import fetch_cifar10c

corruption = ['gaussian_noise', 'motion_blur', 'brightness', 'pixelate']
X, y = fetch_cifar10c(corruption=corruption, severity=5, return_X_y=True)
```

  
**Adversarial CIFAR-10**: `fetch_attack`

  - Load adversarial instances on a ResNet-56 classifier trained on CIFAR-10. Available attacks: [Carlini-Wagner](https://arxiv.org/abs/1608.04644) ('cw') and [SLIDE](https://arxiv.org/abs/1904.13000) ('slide'). Example:
  
```python
from alibi_detect.datasets import fetch_attack

(X_train, y_train), (X_test, y_test) = fetch_attack('cifar10', 'resnet56', 'cw', return_X_y=True)
```

### Tabular

**KDD Cup '99**: `fetch_kdd`

  - Dataset with different types of computer network intrusions. `fetch_kdd` allows you to select a subset of network intrusions as targets or pick only specified features. The original data can be found [here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).

