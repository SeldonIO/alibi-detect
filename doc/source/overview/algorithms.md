# Algorithm Overview

The following tables summarize the advised use cases for the current algorithms. Please consult the method specific pages for a more detailed breakdown of each method. The column *Feature Level* indicates whether the outlier scoring and detection can be done and returned at the feature level, e.g. per pixel for an image.

## Outlier Detection

|Detector|Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|---|---|---|---|---|
|[Isolation Forest](../methods/iforest.ipynb)|✔|✘|✘|✘|✔|✘|✘|
|[Mahalanobis Distance](../methods/mahalanobis.ipynb)|✔|✘|✘|✘|✔|✔|✘|
|[AE](../methods/ae.ipynb)|✔|✔|✘|✘|✘|✘|✔|
|[VAE](../methods/vae.ipynb)|✔|✔|✘|✘|✘|✘|✔|
|[AEGMM](../methods/aegmm.ipynb)|✔|✔|✘|✘|✘|✘|✘|
|[VAEGMM](../methods/vaegmm.ipynb)|✔|✔|✘|✘|✘|✘|✘|
|[Likelihood Ratios](../methods/llr.ipynb)|✔|✔|✔|✘|✔|✘|✔|
|[Prophet](../methods/prophet.ipynb)|✘|✘|✔|✘|✘|✘|✘|
|[Spectral Residual](../methods/sr.ipynb)|✘|✘|✔|✘|✘|✔|✔|
|[Seq2Seq](../methods/seq2seq.ipynb)|✘|✘|✔|✘|✘|✘|✔|

## Adversarial Detection

|Detector|Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|---|---|---|---|---|
|[Adversarial AE](../methods/adversarialae.ipynb)|✔|✔|✘|✘|✘|✘|✘|
|[Model distillation](../methods/modeldistillation.ipynb)|✔|✔|✔|✔|✔|✘|✘|

## Drift Detection

|Detector|Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|---|---|---|---|---|
|[Kolmogorov-Smirnov](../methods/ksdrift.ipynb)|✔|✔|✘|✔|✔|✔|✔|
|[Maximum Mean Discrepancy](../methods/mmddrift.ipynb)|✔|✔|✘|✔|✔|✘|✘|