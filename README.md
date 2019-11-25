[![Build Status](https://travis-ci.com/SeldonIO/alibi-detect.svg?branch=master)](https://travis-ci.com/SeldonIO/alibi-detect)
[![Documentation Status](https://readthedocs.org/projects/alibi-detect/badge/?version=latest)](https://docs.seldon.io/projects/alibi-detect/en/latest/?badge=latest)
![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)
[![PyPI version](https://badge.fury.io/py/alibi-detect.svg)](https://badge.fury.io/py/alibi-detect)
![GitHub Licence](https://img.shields.io/github/license/seldonio/alibi-detect.svg)
[![Slack channel](https://img.shields.io/badge/chat-on%20slack-e51670.svg)](http://seldondev.slack.com/messages/alibi)
---

[alibi-detect](https://github.com/SeldonIO/alibi-detect) is an open source Python library focused on outlier, adversarial and concept drift detection. The package aims to cover both online and offline detectors for tabular data, images and time series. The outlier detection methods should allow the user to identify global, contextual and collective outliers.

*  [Documentation](https://docs.seldon.io/projects/alibi-detect)

## Installation

alibi-detect can be installed from [PyPI](https://pypi.org/project/alibi-detect):
```bash
pip install alibi-detect
```
This will install `alibi-detect` with all its dependencies:
```bash
  creme
  fbprophet
  matplotlib
  numpy
  pandas
  scipy
  scikit-learn
  tensorflow>=2
  tensorflow_probability>=0.8
```

## Supported algorithms

### Outlier Detection

The following table shows the advised use cases for each algorithm. The column *Feature Level* indicates whether the outlier scoring and detection can be done and returned at the feature level, e.g. per pixel for an image:

| Detector              | Tabular | Image | Time Series | Text  | Categorical Features | Online | Feature Level |
| :---                  |  :---:  | :---: |   :---:     | :---: |   :---:              | :---:  | :---:         |
| Isolation Forest      | ✔       | ✘     |  ✘          |  ✘    |  ✔                   |  ✘     |  ✘            |
| Mahalanobis Distance  | ✔       | ✘     |  ✘          |  ✘    |  ✔                   |  ✔     |  ✘            |
| VAE                   | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✔            |
| AEGMM                 | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✘            |
| VAEGMM                | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✘            |
| Prophet               | ✘       | ✘     |  ✔          |  ✘    |  ✘                   |  ✘     |  ✘            |
| Spectral Residual     | ✘       | ✘     |  ✔          |  ✘    |  ✘                   |  ✔     |  ✘            |


 - Isolation Forest ([FT Liu et al., 2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf))
   - [Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/iforest.html)
   - Examples:
     [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_if_kddcup.html)

 - Mahalanobis Distance ([Mahalanobis, 1936](https://insa.nic.in/writereaddata/UpLoadedFiles/PINSA/Vol02_1936_1_Art05.pdf))
   - [Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/mahalanobis.html)
   - Examples:
     [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_mahalanobis_kddcup.html)

 - Variational Auto-Encoder (VAE) ([Kingma et al., 2013](https://arxiv.org/abs/1312.6114))
   - [Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/vae.html)
   - Examples:
     [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_vae_kddcup.html), [CIFAR10](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_vae_cifar10.html)

 - Auto-Encoding Gaussian Mixture Model (AEGMM) ([Zong et al., 2018](https://openreview.net/forum?id=BJJLHbb0-))
   - [Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/aegmm.html)
   - Examples:
     [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_aegmm_kddcup.html)

 - Variational Auto-Encoding Gaussian Mixture Model (VAEGMM)
   - [Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/vaegmm.html)
   - Examples:
     [Network Intrusion](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_aegmm_kddcup.html)
     
 - Prophet Time Series Outlier Detector ([Taylor et al., 2018](https://peerj.com/preprints/3190/))
   - [Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/prophet.html)
   - Examples:
     [Weather Forecast](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_prophet_weather.html)
  
 - Spectral Residual Time Series Outlier Detector ([Ren et al., 2019](https://arxiv.org/abs/1906.03821))
   - [Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/sr.html)
   - Examples:
     [Synthetic Dataset](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_sr_synth.html)


### Adversarial Detection

Advised use cases:

| Detector          | Tabular | Image | Time Series | Text  | Categorical Features | Online | Feature Level |
| :---              |  :---:  | :---: |   :---:     | :---: |   :---:              | :---:  | :---:         |
| Adversarial VAE   | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✘            |


 - Adversarial Variational Auto-Encoder (paper coming soon)
   - [Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/methods/adversarialvae.html)
   - Examples:
     [MNIST](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/ad_advvae_mnist.html)


## Integrations

The integrations folder contains various wrapper tools to allow the alibi-detect algorithms to be used in production machine learning systems with [examples](https://github.com/SeldonIO/alibi-detect/tree/master/integrations/samples/kfserving) on how to deploy outlier and adversarial detectors with [KFServing](https://www.kubeflow.org/docs/components/serving/kfserving/).
