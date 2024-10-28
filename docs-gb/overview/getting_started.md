# Getting Started

### Installation



Alibi Detect can be installed from [PyPI](https://pypi.org/project/alibi-detect/) or [conda-forge](https://conda-forge.org/) by following the instructions below.

#### PyPI

Alibi Detect can be installed from [PyPI](https://pypi.org/project/alibi-detect/) with `pip`. We provide optional dependency buckets for several modules that are large or sometimes tricky to install. Many detectors are supported out of the box with the default install but some detectors require a specific optional dependency installation to use. For instance, the `OutlierProphet` detector requires the prophet installation. Other detectors have a choice of backend. For instance, the `LSDDDrift` detector has a choice of `tensorflow` or `pytorch` backends. The tabs below list the full set of detector functionality provided by each optional dependency.

{% tabs %}
{% tab title="Default" %}
Default installation.

```
pip install alibi-detect
```

The default installation provides out the box support for the following detectors:

* [ChiSquareDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/chisquaredrift.html)
* [CVMDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/cvmdrift.html)
* [FETDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/fetdrift.html)
* [KSDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/ksdrift.html)
* [TabularDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/tabulardrift.html)
* [Mahalanobis](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/mahalanobis.html)
* [SpectralResidual](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/sr.html)
{% endtab %}

{% tab title="Recommended" %}
If you are unsure which detector to use, or wish to have access to as many as possible the recommended installation is:

```
pip install alibi-detect[tensorflow,prophet]
```

If you would rather use `pytorch` backends then you can use:

```
pip install alibi-detect[torch,prophet]
```

However, the following detectors do not have `pytorch` backend support:

* [OutlierAE](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/ae.html)
* [OutlierAEGMM](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/aegmm.html)
* [LLR](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/llr.html)
* [OutlierSeq2Seq](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/seq2seq.html)
* [OutlierVAE](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/vae.html)
* [OutlierVAEGMM](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/vaegmm.html)
* [AdversarialAE](https://docs.seldon.io/projects/alibi-detect/en/stable/ad/methods/adversarialae.html)
* [ModelDistillation](https://docs.seldon.io/projects/alibi-detect/en/stable/ad/methods/modeldistillation.html)

Alternatively you can install all the dependencies using (this will include both `tensorflow` and `pytorch`):

```
pip install alibi-detect[all]
```

Note

If you wish to use the GPU version of PyTorch, or are installing on Windows, it is recommended to [install and test PyTorch](https://pytorch.org/get-started/locally/) prior to installing alibi-detect.

Note

If using torch version 2.0.0 or 2.0.1 along with some versions of tensorflow you may experience hanging depending on the order you import each of these libraries. This is fixed in torch 2.1.0 onwards.
{% endtab %}

{% tab title="PyTorch" %}
Installation with [PyTorch](https://pytorch.org/) backend.

```
pip install alibi-detect[torch]
```

The PyTorch installation is required to use the PyTorch backend for the following detectors:

* [ClassifierDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html)
* [LearnedKernelDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/learnedkerneldrift.html)
* [LSDDDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/lsdddrift.html)
* [LSDDDriftOnline](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinelsdddrift.html)
* [MMDDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html)
* [MMDDriftOnline](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinemmddrift.html)
* [SpotTheDiffDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/spotthediffdrift.html)
* [ContextMMDDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/contextmmddrift.html)

Note

If you wish to use the GPU version of PyTorch, or are installing on Windows, it is recommended to [install and test PyTorch](https://pytorch.org/get-started/locally/) prior to installing alibi-detect.

Note

If using torch version 2.0.0 or 2.0.1 along with some versions of tensorflow you may experience hanging depending on the order you import each of these libraries. This is fixed in torch 2.1.0 onwards.
{% endtab %}

{% tab title="TensorFlow" %}


Installation with [TensorFlow](https://www.tensorflow.org/) backend.

```
pip install alibi-detect[tensorflow]
```

The TensorFlow installation is required to use the TensorFlow backend for the following detectors:

* [ClassifierDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html)
* [LearnedKernelDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/learnedkerneldrift.html)
* [LSDDDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/lsdddrift.html)
* [LSDDDriftOnline](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinelsdddrift.html)
* [MMDDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html)
* [MMDDriftOnline](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinemmddrift.html)
* [SpotTheDiffDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/spotthediffdrift.html)
* [ContextMMDDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/contextmmddrift.html)

The TensorFlow installation is required to use the following detectors:

* [OutlierAE](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/ae.html)
* [OutlierAEGMM](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/aegmm.html)
* [LLR](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/llr.html)
* [OutlierSeq2Seq](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/seq2seq.html)
* [OutlierVAE](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/vae.html)
* [OutlierVAEGMM](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/vaegmm.html)
* [AdversarialAE](https://docs.seldon.io/projects/alibi-detect/en/stable/ad/methods/adversarialae.html)
* [ModelDistillation](https://docs.seldon.io/projects/alibi-detect/en/stable/ad/methods/modeldistillation.html)
{% endtab %}

{% tab title="KeOps" %}
Installation with [KeOps](https://www.kernel-operations.io/) backend.

```
pip install alibi-detect[keops]
```

The KeOps installation is required to use the KeOps backend for the following detectors:

* [MMDDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html)

Note

KeOps requires a C++ compiler compatible with `std=c++11`, for example `g++ >=7` or `clang++ >=8`, and a[Cuda toolkit](https://developer.nvidia.com/cuda-toolkit) installation. For more detailed version requirements and testing instructions for KeOps, see the [KeOps docs](https://www.kernel-operations.io/keops/python/installation.html). **Currently, the KeOps backend is only officially supported on Linux.**
{% endtab %}

{% tab title="Prophet" %}
Installation with [Prophet](https://facebook.github.io/prophet/) support.

```
pip install alibi-detect[prophet]
```

Provides support for the [OutlierProphet](https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/prophet.html) time series outlier detector.
{% endtab %}
{% endtabs %}

#### conda-forge

To install the conda-forge version it is recommended to use [mamba](https://mamba.readthedocs.io/en/stable/), which can be installed to the _base_conda enviroment with:

```
conda install mamba -n base -c conda-forge
```

`mamba` can then be used to install alibi-detect in a conda enviroment:

```
mamba install -c conda-forge alibi-detect
```

## Features

[Alibi Detect](https://github.com/SeldonIO/alibi-detect) is an open source Python library focused on **outlier**, **adversarial** and **drift** detection. The package aims to cover both online and offline detectors for tabular data, text, images and time series. **TensorFlow**, **PyTorch** and (where applicable) [KeOps](https://www.kernel-operations.io/keops/index.html) backends are supported for drift detection. Alibi-Detect does not install these as default. See [installation options](getting\_started.md#installation) for more details.

To get a list of respectively the latest outlier, adversarial and drift detection algorithms, you can type:

```python
import alibi_detect
# View all the Outlier Detection (od) algorithms available
alibi_detect.od.__all__
```

```
['OutlierAEGMM',
 'IForest',
 'Mahalanobis',
 'OutlierAE',
 'OutlierVAE',
 'OutlierVAEGMM',
 'OutlierProphet',
 'OutlierSeq2Seq',
 'SpectralResidual',
 'LLR']
```

```python
# View all the Adversarial Detection (ad) algorithms available
alibi_detect.ad.__all__
```

```
['AdversarialAE',
'ModelDistillation']
```

```python
# View all the Concept Drift (cd) detection algorithms available
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

Summary tables highlighting the practical use cases for all the algorithms can be found [here](algorithms.md).

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

The detectors can be saved or loaded as described in [Saving and loading](saving.md). Finally, we can make predictions on test data and detect outliers or adversarial examples.

```python
preds = od.predict(X_test)
```

The predictions are returned in a dictionary with as keys `meta` and `data`. `meta` contains the detector's metadata while `data` is in itself a dictionary with the actual predictions (and other relevant values). It has either `is_outlier`, `is_adversarial` or `is_drift` (filled with 0's and 1's) as well as optional `instance_score`, `feature_score` or `p_value` as keys with numpy arrays as values.

The exact details will vary slightly from method to method, so we encourage the reader to become familiar with the [types of algorithms supported](algorithms.md) in alibi-detect.
