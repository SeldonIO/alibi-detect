<p align="center">
  <img src="doc/source/_static/Alibi_Detect_Logo.png" alt="Alibi Detect Logo" width="50%">
</p>

[![Build Status](https://travis-ci.com/SeldonIO/alibi-detect.svg?branch=master)](https://travis-ci.com/SeldonIO/alibi-detect)
[![Documentation Status](https://readthedocs.org/projects/alibi-detect/badge/?version=latest)](https://docs.seldon.io/projects/alibi-detect/en/latest/?badge=latest)
![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)
[![PyPI version](https://badge.fury.io/py/alibi-detect.svg)](https://badge.fury.io/py/alibi-detect)
![GitHub Licence](https://img.shields.io/github/license/seldonio/alibi-detect.svg)
[![Slack channel](https://img.shields.io/badge/chat-on%20slack-e51670.svg)](http://seldondev.slack.com/messages/alibi)
---

[alibi-detect](https://github.com/SeldonIO/alibi-detect) is an open source Python library focused on outlier, adversarial and concept drift detection. The package aims to cover both online and offline detectors for tabular data, images and time series. The outlier detection methods should allow the user to identify global, contextual and collective outliers.

*  [Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/)

## Installation

alibi-detect can be installed from [PyPI](https://pypi.org/project/alibi-detect):
```bash
pip install alibi-detect
```
This will install `alibi-detect` with all its dependencies:
```bash
  creme
  fbprophet
  holidays==0.9.11
  matplotlib
  numpy
  pandas
  scipy
  scikit-learn
  tensorflow>=2
  tensorflow_probability>=0.8
```
The save and load functionality for the [Prophet time series outlier detector](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/prophet.html) is currently experiencing [issues in Python 3.6](https://github.com/facebook/prophet/issues/1361) but works in Python 3.7.

## Supported algorithms

The following tables show the advised use cases for each algorithm. The column *Feature Level* indicates whether the detection can be done at the feature level, e.g. per pixel for an image. Check the [algorithm reference list](#reference-list) for more information with links to the documentation and original papers as well as examples for each of the detectors.

### Outlier Detection

| Detector              | Tabular | Image | Time Series | Text  | Categorical Features | Online | Feature Level |
| :---                  |  :---:  | :---: |   :---:     | :---: |   :---:              | :---:  | :---:         |
| Isolation Forest      | ✔       | ✘     |  ✘          |  ✘    |  ✔                   |  ✘     |  ✘            |
| Mahalanobis Distance  | ✔       | ✘     |  ✘          |  ✘    |  ✔                   |  ✔     |  ✘            |
| AE                    | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✔            |
| VAE                   | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✔            |
| AEGMM                 | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✘            |
| VAEGMM                | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✘            |
| Prophet               | ✘       | ✘     |  ✔          |  ✘    |  ✘                   |  ✘     |  ✘            |
| Spectral Residual     | ✘       | ✘     |  ✔          |  ✘    |  ✘                   |  ✔     |  ✔            |
| Seq2Seq               | ✘       | ✘     |  ✔          |  ✘    |  ✘                   |  ✘     |  ✔            |

### Adversarial Detection

| Detector          | Tabular | Image | Time Series | Text  | Categorical Features | Online | Feature Level |
| :---              |  :---:  | :---: |   :---:     | :---: |   :---:              | :---:  | :---:         |
| Adversarial AE    | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✘            |


### Drift Detection

| Detector           | Tabular | Image | Time Series | Text  | Categorical Features | Online | Feature Level |
| :---               |  :---:  | :---: |   :---:     | :---: |   :---:              | :---:  | :---:         |
| Kolmogorov-Smirnov | ✔       | ✔     |  ✔          |  ✘    |  ✘                   |  ✔     |  ✔            |


### Reference List

#### Outlier Detection

- [Isolation Forest](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/iforest.html) ([FT Liu et al., 2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf))
 - Example: [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_if_kddcup.html)

- [Mahalanobis Distance](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/mahalanobis.html) ([Mahalanobis, 1936](https://insa.nic.in/writereaddata/UpLoadedFiles/PINSA/Vol02_1936_1_Art05.pdf))
 - Example: [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_mahalanobis_kddcup.html)

- [Auto-Encoder (AE)](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/ae.html)
 - Example: [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_ae_cifar10.html)

- [Variational Auto-Encoder (VAE)](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/vae.html) ([Kingma et al., 2013](https://arxiv.org/abs/1312.6114))
 - Examples: [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_vae_kddcup.html), [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_vae_cifar10.html)

- [Auto-Encoding Gaussian Mixture Model (AEGMM)](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/aegmm.html) ([Zong et al., 2018](https://openreview.net/forum?id=BJJLHbb0-))
 - Example: [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_aegmm_kddcup.html)

- [Variational Auto-Encoding Gaussian Mixture Model (VAEGMM)](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/vaegmm.html)
 - Example: [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_aegmm_kddcup.html)
     
- [Prophet Time Series Outlier Detector](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/prophet.html) ([Taylor et al., 2018](https://peerj.com/preprints/3190/))
 - Example: [Weather Forecast](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_prophet_weather.html)
  
- [Spectral Residual Time Series Outlier Detector](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/sr.html) ([Ren et al., 2019](https://arxiv.org/abs/1906.03821))
 - Example: [Synthetic Dataset](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_sr_synth.html)

- [Sequence-to-Sequence (Seq2Seq) Outlier Detector](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/seq2seq.html) ([Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf); [Park et al., 2017](https://arxiv.org/pdf/1711.00614.pdf))
 - Examples: [ECG](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_seq2seq_ecg.html), [Synthetic Dataset](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_seq2seq_synth.html)


#### Adversarial Detection

- [Adversarial Auto-Encoder](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/adversarialae.html) ([Vacanti and Van Looveren, 2020](https://arxiv.org/abs/2002.09364))
 - Example: [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/ad_ae_cifar10.html)
     
#### Drift Detection

- [Kolmogorov-Smirnov](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/ksdrift.html)
 - Example: [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_ks_cifar10.html)


## Integrations

The integrations folder contains various wrapper tools to allow the alibi-detect algorithms to be used in production machine learning systems with [examples](https://github.com/SeldonIO/alibi-detect/tree/master/integrations/samples/kfserving) on how to deploy outlier and adversarial detectors with [KFServing](https://www.kubeflow.org/docs/components/serving/kfserving/).

## Citations
If you use alibi-detect in your research, please consider citing it.

BibTeX entry:

```
@software{alibi-detect,
  title = {{Alibi-Detect}: Algorithms for outlier and adversarial instance detection, concept drift and metrics.},
  author = {Van Looveren, Arnaud and Vacanti, Giovanni and Klaise, Janis and Coca, Alexandru},
  url = {https://github.com/SeldonIO/alibi-detect},
  version = {0.3.1},
  date = {2020-02-26},
}
```
