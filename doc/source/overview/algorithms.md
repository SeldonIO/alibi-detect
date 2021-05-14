# Algorithm Overview

The following tables summarize the advised use cases for the current algorithms. Please consult the method specific pages for a more detailed breakdown of each method. The column *Feature Level* indicates whether the outlier scoring and detection can be done and returned at the feature level, e.g. per pixel for an image.

## Outlier Detection

|Detector|Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Isolation Forest](../methods/iforest.ipynb)|✔| | | |✔| | |
|[Mahalanobis Distance](../methods/mahalanobis.ipynb)|✔| | | |✔|✔| |
|[AE](../methods/ae.ipynb)|✔|✔| | | | |✔|
|[VAE](../methods/vae.ipynb)|✔|✔| | | | |✔|
|[AEGMM](../methods/aegmm.ipynb)|✔|✔| | | | | |
|[VAEGMM](../methods/vaegmm.ipynb)|✔|✔| | | | | |
|[Likelihood Ratios](../methods/llr.ipynb)|✔|✔|✔| |✔| |✔|
|[Prophet](../methods/prophet.ipynb)| | |✔| | | | |
|[Spectral Residual](../methods/sr.ipynb)| | |✔| | |✔|✔|
|[Seq2Seq](../methods/seq2seq.ipynb)| | |✔| | | |✔|

## Adversarial Detection

|Detector|Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Adversarial AE](../methods/adversarialae.ipynb)|✔|✔| | | | | |
|[Model distillation](../methods/modeldistillation.ipynb)|✔|✔|✔|✔|✔| | | |

## Drift Detection

|Detector|Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Kolmogorov-Smirnov](../methods/ksdrift.ipynb)|✔|✔| |✔|✔| |✔|
|[Least-Squares Density Difference](../methods/mmddrift.ipynb)|✔|✔| |✔|✔|✔| |
|[Maximum Mean Discrepancy](../methods/mmddrift.ipynb)|✔|✔| |✔|✔|✔| |
|[Chi-Squared](../methods/chisquaredrift.ipynb)|✔| | | |✔| |✔|
|[Mixed-type tabular](../methods/tabulardrift.ipynb)|✔| | | |✔| |✔|
|[Classifier](../methods/classifierdrift.ipynb)|✔|✔|✔|✔|✔| | |
|[Classifier Uncertainty](../methods/modeluncdrift.ipynb)|✔|✔|✔|✔|✔| | |
|[Regressor Uncertainty](../methods/modeluncdrift.ipynb)|✔|✔|✔|✔|✔| | | |

All drift detectors and built-in preprocessing methods support both **PyTorch** and **TensorFlow** backends.
The preprocessing steps include randomly initialized encoders, pretrained text embeddings to detect drift on 
using the [transformers](https://github.com/huggingface/transformers) library and extraction of hidden layers from machine learning models. 
The preprocessing steps allow to detect different types of drift such as covariate and predicted distribution shift.
