# Algorithm Overview

The following tables summarize the advised use cases for the current algorithms. Please consult the method specific pages for a more detailed breakdown of each method. The column *Feature Level* indicates whether the detection can be done and returned at the feature level, e.g. per pixel for an image.

## Outlier Detection

|Detector|Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Isolation Forest](../od/methods/iforest.ipynb)|✔| | | |✔| | |
|[Mahalanobis Distance](../od/methods/mahalanobis.ipynb)|✔| | | |✔|✔| |
|[AE](../od/methods/ae.ipynb)|✔|✔| | | | |✔|
|[VAE](../od/methods/vae.ipynb)|✔|✔| | | | |✔|
|[AEGMM](../od/methods/aegmm.ipynb)|✔|✔| | | | | |
|[VAEGMM](../od/methods/vaegmm.ipynb)|✔|✔| | | | | |
|[Likelihood Ratios](../od/methods/llr.ipynb)|✔|✔|✔| |✔| |✔|
|[Prophet](../od/methods/prophet.ipynb)| | |✔| | | | |
|[Spectral Residual](../od/methods/sr.ipynb)| | |✔| | |✔|✔|
|[Seq2Seq](../od/methods/seq2seq.ipynb)| | |✔| | | |✔|

## Adversarial Detection

|Detector|Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Adversarial AE](../ad/methods/adversarialae.ipynb)|✔|✔| | | | | |
|[Model distillation](../ad/methods/modeldistillation.ipynb)|✔|✔|✔|✔|✔| | | |

## Drift Detection

| Detector                                                          |Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|:------------------------------------------------------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [Kolmogorov-Smirnov](../cd/methods/ksdrift.ipynb)                 |✔|✔| |✔|✔| |✔|
| [Cramér-von Mises](../cd/methods/cvmdrift.ipynb)                  |✔|✔| | | |✔|✔|
| [Fisher's Exact Test](../cd/methods/fetdrift.ipynb)               |✔| | | |✔|✔|✔|
| [Least-Squares Density Difference](../cd/methods/lsdddrift.ipynb) |✔|✔| |✔|✔|✔| |
| [Maximum Mean Discrepancy (MMD)](../cd/methods/mmddrift.ipynb)    |✔|✔| |✔|✔|✔| |
| [Learned Kernel MMD](../cd/methods/learnedkerneldrift.ipynb)      |✔|✔|✔|✔|✔| | |
| [Context-aware MMD](../cd/methods/contextmmddrift.ipynb)          |✔|✔|✔|✔|✔| | | |
| [Chi-Squared](../cd/methods/chisquaredrift.ipynb)                 |✔| | | |✔| |✔|
| [Mixed-type tabular](../cd/methods/tabulardrift.ipynb)            |✔| | | |✔| |✔|
| [Classifier](../cd/methods/classifierdrift.ipynb)                 |✔|✔|✔|✔|✔| | |
| [Spot-the-diff](../cd/methods/spotthediffdrift.ipynb)             |✔|✔|✔|✔|✔| |✔|
| [Classifier Uncertainty](../cd/methods/modeluncdrift.ipynb)       |✔|✔|✔|✔|✔| | |
| [Regressor Uncertainty](../cd/methods/modeluncdrift.ipynb)        |✔|✔|✔|✔|✔| | | |

All drift detectors and built-in preprocessing methods support both **PyTorch** and **TensorFlow** backends.
The preprocessing steps include randomly initialized encoders, pretrained text embeddings to detect drift on 
using the [transformers](https://github.com/huggingface/transformers) library and extraction of hidden layers from machine learning models. 
The preprocessing steps allow to detect different types of drift such as covariate and predicted distribution shift.
