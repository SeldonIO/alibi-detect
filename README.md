[odcd](https://github.com/SeldonIO/odcd) is an open source Python library focused on outlier, adversarial and concept drift detection. The package aims to cover both online and offline detectors for tabular data, images and time series. The outlier detection methods should allow the user to identify global, contextual and collective outliers.

*  [Documentation](https://docs.seldon.io/projects/odcd)

## Installation

odcd can be installed from [PyPI](https://pypi.org/project/odcd):
```bash
pip install odcd
```
This will install `odcd` with all its dependencies:
```bash
  creme
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

 - Isolation Forest ([FT Liu et al., 2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf))
   - [Documentation](https://docs.seldon.io/projects/odcd/en/stable/methods/iforest.html)
   - Examples:
     [Network Intrusion](https://docs.seldon.io/projects/odcd/en/stable/examples/od_if_kddcup.html)
     
 - Mahalanobis Distance ([Mahalanobis, 1936](https://insa.nic.in/writereaddata/UpLoadedFiles/PINSA/Vol02_1936_1_Art05.pdf))
   - [Documentation](https://docs.seldon.io/projects/odcd/en/stable/methods/mahalanobis.html)
   - Examples:
     [Network Intrusion](https://docs.seldon.io/projects/odcd/en/stable/examples/od_mahalanobis_kddcup.html)

 - Variational Auto-Encoder (VAE) ([Kingma et al., 2013](https://arxiv.org/abs/1312.6114))
   - [Documentation](https://docs.seldon.io/projects/odcd/en/stable/methods/vae.html)
   - Examples:
     [Network Intrusion](https://docs.seldon.io/projects/odcd/en/stable/examples/od_vae_kddcup.html), [CIFAR10](https://docs.seldon.io/projects/odcd/en/stable/examples/od_vae_cifar10.html)
     
 - Auto-Encoding Gaussian Mixture Model (AEGMM) ([Zong et al., 2018](https://openreview.net/forum?id=BJJLHbb0-))
   - [Documentation](https://docs.seldon.io/projects/odcd/en/stable/methods/aegmm.html)
   - Examples:
     [Network Intrusion](https://docs.seldon.io/projects/odcd/en/stable/examples/od_aegmm_kddcup.html)

 - Variational Auto-Encoding Gaussian Mixture Model (VAEGMM)
   - [Documentation](https://docs.seldon.io/projects/odcd/en/stable/methods/vaegmm.html)
   - Examples:
     [Network Intrusion](https://docs.seldon.io/projects/odcd/en/stable/examples/od_aegmm_kddcup.html)
     
Please check the documentation 

The following table shows the advised use cases for each algorithm. The column *Feature Level* indicates whether the outlier scoring and detection can be done and returned at the feature level, e.g. per pixel for an image:

| Detector              | Tabular | Image | Time Series | Text  | Categorical Features | Online | Feature Level |
| :---                  |  :---:  | :---: |   :---:     | :---: |   :---:              | :---:  | :---:         |
| Isolation Forest      | ✔       | ✘     |  ✘          |  ✘    |  ✔                   |  ✘     |  ✘            |
| Mahalanobis Distance  | ✔       | ✘     |  ✘          |  ✘    |  ✔                   |  ✔     |  ✘            |
| VAE                   | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✔            |
| AEGMM                 | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✘            |
| VAEGMM                | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✘            |


### Adversarial Detection

 - Adversarial Variational Auto-Encoder
   - [Documentation](https://docs.seldon.io/projects/odcd/en/stable/methods/adversarialvae.html)
   - Examples:
     [MNIST](https://docs.seldon.io/projects/odcd/en/stable/examples/ad_advvae_mnist.html)

Advised use cases:

| Detector          | Tabular | Image | Time Series | Text  | Categorical Features | Online | Feature Level |
| :---              |  :---:  | :---: |   :---:     | :---: |   :---:              | :---:  | :---:         |
| Adversarial VAE   | ✔       | ✔     |  ✘          |  ✘    |  ✘                   |  ✘     |  ✘            |
