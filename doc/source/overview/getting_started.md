# Getting Started

## Installation

Alibi Detect can be installed from [PyPI](https://pypi.org/project/alibi-detect/) or [conda-forge](https://conda-forge.org/)
by following the instructions below.


``````{dropdown} Install via PyPI

```{div} sd-mb-3
- Alibi Detect can be installed from [PyPI](https://pypi.org/project/alibi-detect/) with `pip`:
```

`````{tab-set}

````{tab-item} Standard
:sync: label-standard
:class-label: sd-pt-0

```{div} sd-mb-1
Installation with default [TensorFlow](https://www.tensorflow.org/) backend.
```

```bash
pip install alibi-detect
```
````

````{tab-item} PyTorch
:sync: label-torch
:class-label: sd-pt-0

```{div} sd-mb-1
Installation with [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) backends.
```

```bash
pip install alibi-detect[torch]
```

```{note}
If you wish to use the GPU version of PyTorch, or are installing on Windows, it is recommended to [install and test PyTorch](https://pytorch.org/get-started/locally/) 
prior to installing alibi-detect.
```
````

````{tab-item} Prophet
:class-label: sd-pt-0

```{div} sd-mb-1
Installation with the [Prophet](../od/methods/prophet.ipynb) time series outlier detector enabled.
```

```bash
pip install alibi-detect[prophet]
```
````

`````
``````


``````{dropdown} Install via conda-forge

```{div} sd-mb-3
- To install the conda-forge version it is recommended to use [mamba](https://mamba.readthedocs.io/en/latest/), 
which can be installed to the *base* conda enviroment with:
```

```bash
conda install mamba -n base -c conda-forge
```

```{div} sd-mb-3
- `mamba` can then be used to install alibi-detect in a conda enviroment:
```

`````{tab-set}

````{tab-item} Standard
:sync: label-standard
:class-label: sd-pt-0

Installation with default [TensorFlow](https://www.tensorflow.org/) backend.

```bash
mamba install -c conda-forge alibi-detect
```
````

````{tab-item} PyTorch
:sync: label-torch
:class-label: sd-pt-0

Installation with [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) backends.

```bash
mamba install -c conda-forge alibi-detect pytorch
```

```{note}
If you wish to use the GPU version of PyTorch, or are installing on Windows, it is recommended to [install and test PyTorch](https://pytorch.org/get-started/locally/) 
prior to installing alibi-detect.
```
````

`````
``````


## Features

[Alibi Detect](https://github.com/SeldonIO/alibi-detect) is an open source Python library focused on 
**outlier**, **adversarial** and **drift** detection. The package aims to cover both 
online and offline detectors for tabular data, text, images and time series. 
Both **TensorFlow** and **PyTorch** backends are supported for drift detection. Alibi Detect does however 
not install PyTorch for you. Check the [PyTorch docs](https://pytorch.org/) how to do this.

To get a list of respectively the latest outlier, adversarial and drift detection algorithms, you can type:

```python
import alibi_detect
alibi_detect.od.__all__
```

```
['OutlierAEGMM',
 'IForest',
 'Mahalanobis',
 'OutlierAE',
 'OutlierVAE',
 'OutlierVAEGMM',
 'OutlierProphet',  # requires prophet: pip install alibi-detect[prophet]
 'OutlierSeq2Seq',
 'SpectralResidual',
 'LLR']
```

```python
alibi_detect.ad.__all__
```

```
['AdversarialAE',
'ModelDistillation']
```

```python
alibi_detect.cd.__all__
```

```
['ChiSquareDrift',
 'ClassifierDrift',
 'ClassifierUncertaintyDrift',
 'ContextMMDDrift',
 'CVMDrift',
 'FETDrift',
 'KSDrift',
 'LearnedKernelDrift',
 'LSDDDrift',
 'LSDDDriftOnline',
 'MMDDrift',
 'MMDDriftOnline',
 'RegressorUncertaintyDrift',
 'SpotTheDiffDrift',
 'TabularDrift']
```

Summary tables highlighting the practical use cases for all the algorithms can be found [here](../overview/algorithms.md).

For detailed information on the **outlier detectors**:

* [Isolation Forest](../od/methods/iforest.ipynb)

* [Mahalanobis Distance](../od/methods/mahalanobis.ipynb)
    
* [Auto-Encoder (AE)](../od/methods/ae.ipynb)

* [Variational Auto-Encoder (VAE)](../od/methods/vae.ipynb)

* [Auto-Encoding Gaussian Mixture Model (AEGMM)](../od/methods/aegmm.ipynb)

* [Variational Auto-Encoding Gaussian Mixture Model (VAEGMM)](../od/methods/vaegmm.ipynb)

* [Likelihood Ratios](../od/methods/llr.ipynb)
    
* [Prophet Detector](../od/methods/prophet.ipynb)
    
* [Spectral Residual Detector](../od/methods/sr.ipynb)
    
* [Sequence-to-Sequence (Seq2Seq) Detector](../od/methods/seq2seq.ipynb)

Similar for **adversarial detection**:

* [Adversarial AE Detector](../ad/methods/adversarialae.ipynb)
  
* [Model Distillation Detector](../ad/methods/modeldistillation.ipynb)

And **data drift**:

* [Chi-Squared Drift Detector](../cd/methods/chisquaredrift.ipynb)

* [Classifier Drift Detector](../cd/methods/classifierdrift.ipynb)

* [Classifier and Regressor Uncertainty Drift Detectors](../cd/methods/modeluncdrift.ipynb)

* [Context-aware Drift Detector](../cd/methods/contextmmddrift.ipynb)

* [Cramér-von Mises Drift Detector](../cd/methods/cvmdrift.ipynb)

* [Fisher's Exact Test Drift Detector](../cd/methods/fetdrift.ipynb)

* [Kolmogorov-Smirnov Drift Detector](../cd/methods/ksdrift.ipynb)

* [Learned Kernel MMD Drift Detector](../cd/methods/learnedkerneldrift.ipynb)

* [Least-Squares Density Difference Drift Detector](../cd/methods/lsdddrift.ipynb)

* [Online Least-Squares Density Difference Drift Detector](../cd/methods/onlinelsdddrift.ipynb)

* [Maximum Mean Discrepancy (MMD) Drift Detector](../cd/methods/mmddrift.ipynb)

* [Online Maximum Mean Discrepancy Drift Detector](../cd/methods/onlinemmddrift.ipynb)

* [Spot-the-diff Drift Detector](../cd/methods/spotthediffdrift.ipynb)

* [Mixed-type Tabular Data Drift Detector](../cd/methods/tabulardrift.ipynb)

## Basic Usage

We will use the [VAE outlier detector](../od/methods/vae.ipynb) to illustrate the usage of outlier and adversarial detectors in alibi-detect.

First, we import the detector:

```python
from alibi_detect.od import OutlierVAE
```

Then we initialize it by passing it the necessary arguments:

```python
od = OutlierVAE(
    threshold=0.1,
    encoder_net=encoder_net,
    decoder_net=decoder_net,
    latent_dim=1024
)
```

Some detectors require an additional `.fit` step using training data:

```python
od.fit(X_train)
```

The detectors can be saved or loaded as described in [Saving and loading](saving.md). 
Finally, we can make predictions on test data and detect outliers or adversarial examples.

```python
preds = od.predict(X_test)
```

The predictions are returned in a dictionary with as keys `meta` and `data`. `meta` contains the detector's metadata while `data` is in itself a dictionary with the actual predictions (and other relevant values). It has either `is_outlier`, `is_adversarial` or `is_drift` (filled with 0's and 1's) as well as optional `instance_score`, `feature_score` or `p_value` as keys with numpy arrays as values.

The exact details will vary slightly from method to method, so we encourage the reader to become
familiar with the [types of algorithms supported](../overview/algorithms.md) in alibi-detect.
