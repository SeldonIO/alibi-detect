# Getting Started

## Installation

alibi-detect can be installed from [PyPI](https://pypi.org/project/alibi-detect/):

```bash
pip install alibi-detect
```

## Features

alibi-detect is a Python package focused on outlier, adversarial and concept drift detection. The package aims to cover both online and offline detectors for tabular data, text, images and time series. The outlier detection methods should allow the user to identify global, contextual and collective outliers.

To get a list of respectively the latest outlier and adversarial detection algorithms, you can type:

```python
import alibi_detect
alibi_detect.od.__all__
```

```
['IForest',
 'Mahalanobis',
 'OutlierAEGMM',
 'OutlierVAE',
 'OutlierVAEGMM']
```

```python
alibi_detect.ad.__all__
```

```
['AdversarialVAE']
```

For detailed information on the methods:

* [Overview of available methods](../overview/algorithms.md)

    * [Isolation Forest Outlier Detector](../methods/iforest.ipynb)

    * [Mahalanobis Distance Outlier Detector](../methods/mahalanobis.ipynb)

    * [Variational Auto-Encoder (VAE) Outlier Detector](../methods/vae.ipynb)

    * [Auto-Encoding Gaussian Mixture Model (AEGMM) Outlier Detector](../methods/aegmm.ipynb)

    * [Variational Auto-Encoding Gaussian Mixture Model (VAEGMM) Outlier Detector](../methods/vaegmm.ipynb)

    * [Adversarial VAE Detector](../methods/adversarialvae.ipynb)

## Basic Usage

We will use the [VAE outlier detector](../methods/vae.ipynb) to illustrate the usage of outlier and adversarial detectors in alibi-detect.

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

The detectors can be saved or loaded as follows:

```python
from alibi_detect.utils.saving import save_detector, load_detector

filepath = './my_detector/'
save_detector(od, filepath)
od = load_detector(filepath)
```

Finally, we can make predictions on test data and detect outliers or adversarial examples.

```python
preds = od.predict(X_test)
```

The predictions are returned in a dictionary with as keys `meta` and `data`. `meta` contains the detector's metadata while `data` is in itself a dictionary with the actual predictions. It has either `is_outlier` or `is_adversarial` (filled with 0's and 1's) as well as `instance_score` and `feature_score` as keys with numpy arrays as values.

The exact details will vary slightly from method to method, so we encourage the reader to become
familiar with the [types of algorithms supported](../overview/algorithms.md) in alibi-detect.