<p align="center">
  <img src="https://raw.githubusercontent.com/SeldonIO/alibi-detect/master/doc/source/_static/Alibi_Detect_Logo_rgb.png" alt="Alibi Detect Logo" width="50%">
</p>

<!--- BADGES: START --->

<!---
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)
[![PyPI version](https://badge.fury.io/py/alibi-detect.svg)](https://badge.fury.io/py/alibi-detect)
--->
[![Build Status](https://github.com/SeldonIO/alibi-detect/workflows/CI/badge.svg?branch=master)][#build-status]
[![Documentation Status](https://readthedocs.org/projects/alibi-detect/badge/?version=latest)][#docs-package]
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)
[![PyPI - Package Version](https://img.shields.io/pypi/v/alibi-detect?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/alibi-detect?logo=anaconda&style=flat&color=orange)][#conda-forge-package]
[![GitHub - License](https://img.shields.io/github/license/SeldonIO/alibi-detect?logo=github&style=flat&color=green)][#github-license]
[![Slack channel](https://img.shields.io/badge/chat-on%20slack-e51670.svg)][#slack-channel]
<!---[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/alibi-detect?logo=pypi&style=flat&color=blue)][#pypi-package]--->
<!--- TODO switch to above auto Python version on new release (PyPi needs updating with classiferies in setup.py classifiers) --->

<!--- Hide platform for now as platform agnostic --->
<!--- [![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/alibi-detect?logo=anaconda&style=flat)][#conda-forge-package]--->

[#github-license]: https://github.com/SeldonIO/alibi-detect/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/alibi-detect/
[#conda-forge-package]: https://anaconda.org/conda-forge/alibi-detect
[#docs-package]: https://docs.seldon.io/projects/alibi-detect/en/latest/
[#build-status]: https://github.com/SeldonIO/alibi-detect/actions?query=workflow%3A%22CI%22
[#slack-channel]: https://join.slack.com/t/seldondev/shared_invite/zt-vejg6ttd-ksZiQs3O_HOtPQsen_labg
<!--- BADGES: END --->
---

[Alibi Detect](https://github.com/SeldonIO/alibi-detect) is an open source Python library focused on **outlier**, **adversarial** and **drift** detection. The package aims to cover both online and offline detectors for tabular data, text, images and time series. Both **TensorFlow** and **PyTorch** backends are supported for drift detection.
*  [Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/)

For more background on the importance of monitoring outliers and distributions in a production setting, check out [this talk](https://slideslive.com/38931758/monitoring-and-explainability-of-models-in-production?ref=speaker-37384-latest) from the *Challenges in Deploying and Monitoring Machine Learning Systems* ICML 2020 workshop, based on the paper [Monitoring and explainability of models in production](https://arxiv.org/abs/2007.06299) and referencing Alibi Detect.

For a thorough introduction to drift detection, check out [Protecting Your Machine Learning Against Drift: An Introduction](https://youtu.be/tL5sEaQha5o). The talk covers what drift is and why it pays to detect it, the different types of drift, how it can be detected in a principled manner and also describes the anatomy of a drift detector.


## Table of Contents


- [Installation and Usage](#installation-and-usage)
  - [With pip](#with-pip)
  - [With conda](#with-conda)
  - [Usage](#usage)   
- [Supported Algorithms](#supported-algorithms)
  - [Outlier Detection](#outlier-detection)
  - [Adversarial Detection](#adversarial-detection)
  - [Drift Detection](#drift-detection)
    - [TensorFlow and PyTorch support](#tensorflow-and-pytorch-support)
    - [Built-in preprocessing steps](#built-in-preprocessing-steps)
  - [Reference List](#reference-list)
    - [Outlier Detection](#outlier-detection-1)
    - [Adversarial Detection](#adversarial-detection-1)
    - [Drift Detection](#drift-detection-1)
- [Datasets](#datasets)
  - [Sequential Data and Time Series](#sequential-data-and-time-series)
  - [Images](#images)
  - [Tabular](#tabular)
- [Models](#models)
- [Integrations](#integrations)
- [Citations](#citations)



## Installation and Usage

The package, `alibi-detect` can be installed from:

- PyPI or GitHub source (with `pip`)
- Anaconda (with `conda`/`mamba`)

### With pip

- alibi-detect can be installed from [PyPI](https://pypi.org/project/alibi-detect):

   ```bash
   pip install alibi-detect
   ```
   
- Alternatively, the development version can be installed:

   ```bash
   pip install git+https://github.com/SeldonIO/alibi-detect.git
   ```

- To install with the PyTorch backend (in addition to the default TensorFlow backend):
  ```bash
  pip install alibi-detect[torch]
  ```

- To use the `Prophet` time series outlier detector:

   ```bash
   pip install alibi-detect[prophet]
   ```

### With conda

To install from [conda-forge](https://conda-forge.org/) it is recommended to use [mamba](https://mamba.readthedocs.io/en/latest/), 
which can be installed to the *base* conda enviroment with:

```bash
conda install mamba -n base -c conda-forge
```

- To install alibi-detect with the default TensorFlow backend:

  ```bash
  mamba install -c conda-forge alibi-detect
  ```

- To install with the PyTorch backend:

  ```bash
  mamba install -c conda-forge alibi-detect pytorch
  ```

### Usage
We will use the [VAE outlier detector](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/vae.html) to illustrate the API.

```python
from alibi_detect.od import OutlierVAE
from alibi_detect.utils import save_detector, load_detector

# initialize and fit detector
od = OutlierVAE(threshold=0.1, encoder_net=encoder_net, decoder_net=decoder_net, latent_dim=1024)
od.fit(x_train)

# make predictions
preds = od.predict(x_test)

# save and load detectors
filepath = './my_detector/'
save_detector(od, filepath)
od = load_detector(filepath)
```

The predictions are returned in a dictionary with as keys `meta` and `data`. `meta` contains the detector's metadata while `data` is in itself a dictionary with the actual predictions. It contains the outlier, adversarial or drift scores and thresholds as well as the predictions whether instances are e.g. outliers or not. The exact details can vary slightly from method to method, so we encourage the reader to become familiar with the [types of algorithms supported](https://docs.seldon.io/projects/alibi-detect/en/latest/overview/algorithms.html).

## Supported Algorithms

The following tables show the advised use cases for each algorithm. The column *Feature Level* indicates whether the detection can be done at the feature level, e.g. per pixel for an image. Check the [algorithm reference list](#reference-list) for more information with links to the documentation and original papers as well as examples for each of the detectors.

### Outlier Detection

| Detector             | Tabular | Image | Time Series | Text | Categorical Features | Online | Feature Level |
|:---------------------|:-------:|:-----:|:-----------:|:----:|:--------------------:|:------:|:-------------:|
| Isolation Forest     |    ✔    |       |             |      |          ✔           |        |               |
| Mahalanobis Distance |    ✔    |       |             |      |          ✔           |   ✔    |               |
| AE                   |    ✔    |   ✔   |             |      |                      |        |       ✔       |
| VAE                  |    ✔    |   ✔   |             |      |                      |        |       ✔       |
| AEGMM                |    ✔    |   ✔   |             |      |                      |        |               |
| VAEGMM               |    ✔    |   ✔   |             |      |                      |        |               |
| Likelihood Ratios    |    ✔    |   ✔   |      ✔      |      |          ✔           |        |       ✔       |
| Prophet              |         |       |      ✔      |      |                      |        |               |
| Spectral Residual    |         |       |      ✔      |      |                      |   ✔    |       ✔       |
| Seq2Seq              |         |       |      ✔      |      |                      |        |       ✔       |

### Adversarial Detection

| Detector           | Tabular | Image | Time Series | Text | Categorical Features | Online | Feature Level |
| :---               |  :---:  | :---: |:-----------:|:----:|:--------------------:|:------:|:-------------:|
| Adversarial AE     | ✔       | ✔     |             |      |                      |        |               |
| Model distillation | ✔       | ✔     |      ✔      |  ✔   |          ✔           |        |               |


### Drift Detection

| Detector                         | Tabular | Image | Time Series | Text  | Categorical Features | Online | Feature Level |
|:---------------------------------|  :---:  | :---: |   :---:     | :---: |   :---:              | :---:  | :---:         |
| Kolmogorov-Smirnov               | ✔       | ✔     |             | ✔     | ✔                    |        | ✔             |
| Cramér-von Mises                 | ✔       | ✔     |             |       |                      | ✔      | ✔             |
| Fisher's Exact Test              | ✔       |       |             |       | ✔                    | ✔      | ✔             |
| Maximum Mean Discrepancy (MMD)   | ✔       | ✔     |             | ✔     | ✔                    | ✔      |               |
| Learned Kernel MMD               | ✔       | ✔     |             | ✔     | ✔                    |        |               |
| Context-aware MMD                | ✔       | ✔     |  ✔          | ✔     | ✔                    |        |               |
| Least-Squares Density Difference | ✔       | ✔     |             | ✔     | ✔                    | ✔      |               |
| Chi-Squared                      | ✔       |       |             |       | ✔                    |        | ✔             |
| Mixed-type tabular data          | ✔       |       |             |       | ✔                    |        | ✔             |
| Classifier                       | ✔       | ✔     |  ✔          | ✔     | ✔                    |        |               |
| Spot-the-diff                    | ✔       | ✔     |  ✔          | ✔     | ✔                    |        | ✔             |
| Classifier Uncertainty           | ✔       | ✔     |  ✔          | ✔     | ✔                    |        |               |
| Regressor Uncertainty            | ✔       | ✔     |  ✔          | ✔     | ✔                    |        |               |

#### TensorFlow and PyTorch support

The drift detectors support TensorFlow and PyTorch backends. Alibi Detect does however not install PyTorch for you. 
Check the [PyTorch docs](https://pytorch.org/) how to do this. Example:

```python
from alibi_detect.cd import MMDDrift

cd = MMDDrift(x_ref, backend='tensorflow', p_val=.05)
preds = cd.predict(x)
```

The same detector in PyTorch:

```python
cd = MMDDrift(x_ref, backend='pytorch', p_val=.05)
preds = cd.predict(x)
```

#### Built-in preprocessing steps

Alibi Detect also comes with various preprocessing steps such as randomly initialized encoders, pretrained text
embeddings to detect drift on using the [transformers](https://github.com/huggingface/transformers) library and 
extraction of hidden layers from machine learning models. This allows to detect different types of drift such as 
**covariate and predicted distribution shift**. The preprocessing steps are again supported in TensorFlow and PyTorch.

```python
from alibi_detect.cd.tensorflow import HiddenOutput, preprocess_drift

model = # TensorFlow model; tf.keras.Model or tf.keras.Sequential
preprocess_fn = partial(preprocess_drift, model=HiddenOutput(model, layer=-1), batch_size=128)
cd = MMDDrift(x_ref, backend='tensorflow', p_val=.05, preprocess_fn=preprocess_fn)
preds = cd.predict(x)
```

Check the example notebooks (e.g. [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_mmd_cifar10.html), [movie reviews](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_text_imdb.html)) for more details.

### Reference List

#### Outlier Detection

- [Isolation Forest](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/iforest.html) ([FT Liu et al., 2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf))
   - Example: [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_if_kddcup.html)

- [Mahalanobis Distance](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/mahalanobis.html) ([Mahalanobis, 1936](https://insa.nic.in/writereaddata/UpLoadedFiles/PINSA/Vol02_1936_1_Art05.pdf))
   - Example: [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_mahalanobis_kddcup.html)

- [Auto-Encoder (AE)](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html)
   - Example: [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_ae_cifar10.html)

- [Variational Auto-Encoder (VAE)](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/vae.html) ([Kingma et al., 2013](https://arxiv.org/abs/1312.6114))
   - Examples: [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_vae_kddcup.html), [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_vae_cifar10.html)

- [Auto-Encoding Gaussian Mixture Model (AEGMM)](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/aegmm.html) ([Zong et al., 2018](https://openreview.net/forum?id=BJJLHbb0-))
   - Example: [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_aegmm_kddcup.html)

- [Variational Auto-Encoding Gaussian Mixture Model (VAEGMM)](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/vaegmm.html)
   - Example: [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_aegmm_kddcup.html)
     
- [Likelihood Ratios](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/llr.html) ([Ren et al., 2019](https://arxiv.org/abs/1906.02845))
   - Examples: [Genome](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_llr_genome.html), [Fashion-MNIST vs. MNIST](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_llr_mnist.html)

- [Prophet Time Series Outlier Detector](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/prophet.html) ([Taylor et al., 2018](https://peerj.com/preprints/3190/))
   - Example: [Weather Forecast](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_prophet_weather.html)
  
- [Spectral Residual Time Series Outlier Detector](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/sr.html) ([Ren et al., 2019](https://arxiv.org/abs/1906.03821))
   - Example: [Synthetic Dataset](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_sr_synth.html)

- [Sequence-to-Sequence (Seq2Seq) Outlier Detector](https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/seq2seq.html) ([Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf); [Park et al., 2017](https://arxiv.org/pdf/1711.00614.pdf))
   - Examples: [ECG](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_seq2seq_ecg.html), [Synthetic Dataset](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_seq2seq_synth.html)
  
#### Adversarial Detection

- [Adversarial Auto-Encoder](https://docs.seldon.io/projects/alibi-detect/en/latest/ad/methods/adversarialae.html) ([Vacanti and Van Looveren, 2020](https://arxiv.org/abs/2002.09364))
   - Example: [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/ad_ae_cifar10.html)

- [Model distillation](https://docs.seldon.io/projects/alibi-detect/en/latest/ad/methods/modeldistillation.html) 
   - Example: [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_distillation_cifar10.html)
     
#### Drift Detection

- [Kolmogorov-Smirnov](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/ksdrift.html)
   - Example: [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_ks_cifar10.html), [molecular graphs](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_mol.html), [movie reviews](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_text_imdb.html)

- [Cramér-von Mises](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/cvmdrift.html)
  - Example: [Penguins](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_supervised_penguins.html)

- [Fisher's Exact Test](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/fetdrift.html)
  - Example: [Penguins](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_supervised_penguins.html)

- [Least-Squares Density Difference](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/lsdddrift.html) ([Bu et al, 2016](https://alippi.faculty.polimi.it/articoli/A%20Pdf%20free%20Change%20Detection%20Test%20Based%20on%20Density%20Difference%20Estimation.pdf))

- [Maximum Mean Discrepancy](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/mmddrift.html) ([Gretton et al, 2012](http://jmlr.csail.mit.edu/papers/v13/gretton12a.html))
   - Example: [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_mmd_cifar10.html), [molecular graphs](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_mol.html), [movie reviews](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_text_imdb.html), [Amazon reviews](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_text_amazon.html)

- [Learned Kernel MMD](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/learnedkerneldrift.html) ([Liu et al, 2020](https://arxiv.org/abs/2002.09116))
  - Example: [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_clf_cifar10.html)

- [Context-aware MMD](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/contextmmddrift.html) ([Cobb and Van Looveren, 2022](https://arxiv.org/abs/2203.08644))
  - Example: [ECG](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_context_ecg.html), [news topics](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_context_20newsgroup.html)

- [Chi-Squared](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/chisquaredrift.html)
   - Example: [Income Prediction](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_chi2ks_adult.html)

- [Mixed-type tabular data](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/tabulardrift.html)
   - Example: [Income Prediction](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_chi2ks_adult.html)

- [Classifier](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/classifierdrift.html) ([Lopez-Paz and Oquab, 2017](https://openreview.net/forum?id=SJkXfE5xx))
   - Example: [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_clf_cifar10.html), [Amazon reviews](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_text_amazon.html)

- [Spot-the-diff](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/spotthediffdrift.html) (adaptation of [Jitkrittum et al, 2016](https://arxiv.org/abs/1605.06796))
  - Example [MNIST and Wine quality](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/spot_the_diff_mnist_win.html)

- [Classifier and Regressor Uncertainty](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/modeluncdrift.html)
   - Example: [CIFAR10 and Wine](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_model_unc_cifar10_wine.html), [molecular graphs](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_mol.html)

- [Online Maximum Mean Discrepancy](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/onlinemmddrift.html)
  - Example: [Wine Quality](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_online_wine.html), [Camelyon medical imaging](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_online_camelyon.html)
  
- [Online Least-Squares Density Difference](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/onlinemmddrift.html) ([Bu et al, 2017](https://ieeexplore.ieee.org/abstract/document/7890493))
  - Example: [Wine Quality](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_online_wine.html)


## Datasets

The package also contains functionality in `alibi_detect.datasets` to easily fetch a number of datasets for different modalities. For each dataset either the data and labels or a *Bunch* object with the data, labels and optional metadata are returned. Example:

```python
from alibi_detect.datasets import fetch_ecg

(X_train, y_train), (X_test, y_test) = fetch_ecg(return_X_y=True)
```

### Sequential Data and Time Series

- **Genome Dataset**: `fetch_genome`
  - Bacteria genomics dataset for out-of-distribution detection, released as part of [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845). From the original *TL;DR*: *The dataset contains genomic sequences of 250 base pairs from 10 in-distribution bacteria classes for training, 60 OOD bacteria classes for validation, and another 60 different OOD bacteria classes for test*. There are respectively 1, 7 and again 7 million sequences in the training, validation and test sets. For detailed info on the dataset check the [README](https://storage.cloud.google.com/seldon-datasets/genome/readme.docx?organizationId=156002945562).
  
  ```python
  from alibi_detect.datasets import fetch_genome
  
  (X_train, y_train), (X_val, y_val), (X_test, y_test) = fetch_genome(return_X_y=True)
  ```

- **ECG 5000**: `fetch_ecg`
  - 5000 ECG's, originally obtained from [Physionet](https://archive.physionet.org/cgi-bin/atm/ATM).

- **NAB**: `fetch_nab`
  - Any univariate time series in a DataFrame from the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB). A list with the available time series can be retrieved using `alibi_detect.datasets.get_list_nab()`.


### Images

- **CIFAR-10-C**: `fetch_cifar10c`
  - CIFAR-10-C ([Hendrycks & Dietterich, 2019](https://arxiv.org/abs/1903.12261)) contains the test set of CIFAR-10, but corrupted and perturbed by various types of noise, blur, brightness etc. at different levels of severity, leading to a gradual decline in a classification model's performance trained on CIFAR-10. `fetch_cifar10c` allows you to pick any severity level or corruption type. The list with available corruption types can be retrieved with `alibi_detect.datasets.corruption_types_cifar10c()`. The dataset can be used in research on robustness and drift. The original data can be found [here](https://zenodo.org/record/2535967#.XnAM2nX7RNw). Example:
  
  ```python
  from alibi_detect.datasets import fetch_cifar10c
  
  corruption = ['gaussian_noise', 'motion_blur', 'brightness', 'pixelate']
  X, y = fetch_cifar10c(corruption=corruption, severity=5, return_X_y=True)
  ```
  
- **Adversarial CIFAR-10**: `fetch_attack`
  - Load adversarial instances on a ResNet-56 classifier trained on CIFAR-10. Available attacks: [Carlini-Wagner](https://arxiv.org/abs/1608.04644) ('cw') and [SLIDE](https://arxiv.org/abs/1904.13000) ('slide'). Example:
  
  ```python
  from alibi_detect.datasets import fetch_attack
  
  (X_train, y_train), (X_test, y_test) = fetch_attack('cifar10', 'resnet56', 'cw', return_X_y=True)
  ```

### Tabular

- **KDD Cup '99**: `fetch_kdd`
  - Dataset with different types of computer network intrusions. `fetch_kdd` allows you to select a subset of network intrusions as targets or pick only specified features. The original data can be found [here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).


## Models

Models and/or building blocks that can be useful outside of outlier, adversarial or drift detection can be found under `alibi_detect.models`. Main implementations:

- [PixelCNN++](https://arxiv.org/abs/1701.05517): `alibi_detect.models.pixelcnn.PixelCNN`

- Variational Autoencoder: `alibi_detect.models.autoencoder.VAE`

- Sequence-to-sequence model: `alibi_detect.models.autoencoder.Seq2Seq`

- ResNet: `alibi_detect.models.resnet`
  - Pre-trained ResNet-20/32/44 models on CIFAR-10 can be found on our [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/seldon-models/alibi-detect/classifier/cifar10/?organizationId=156002945562&project=seldon-pub) and can be fetched as follows:

  ```python
  from alibi_detect.utils.fetching import fetch_tf_model
  
  model = fetch_tf_model('cifar10', 'resnet32')
  ```

## Integrations

Alibi-detect is integrated in the open source machine learning model deployment platform [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/index.html) and model serving framework [KFServing](https://github.com/kubeflow/kfserving).

- **Seldon Core**: [outlier](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/outlier_detection.html) and [drift](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/drift_detection.html) detection worked examples.

- **KFServing**: [outlier](https://github.com/kubeflow/kfserving/tree/master/docs/samples/outlier-detection/alibi-detect/cifar10) and [drift](https://github.com/kubeflow/kfserving/tree/master/docs/samples/drift-detection/alibi-detect/cifar10) detection examples.


## Citations
If you use alibi-detect in your research, please consider citing it.

BibTeX entry:

```
@software{alibi-detect,
  title = {Alibi Detect: Algorithms for outlier, adversarial and drift detection},
  author = {Van Looveren, Arnaud and Klaise, Janis and Vacanti, Giovanni and Cobb, Oliver and Scillitoe, Ashley and Samoilescu, Robert},
  url = {https://github.com/SeldonIO/alibi-detect},
  version = {0.9.0},
  date = {2022-03-17},
  year = {2019}
}
```
