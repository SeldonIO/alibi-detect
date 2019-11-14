# Algorithm Overview

The following tables summarize the advised use cases for the current algorithms. Please consult the method specific pages for a more detailed breakdown of each method. The column *Feature Level* indicates whether the outlier scoring and detection can be done and returned at the feature level, e.g. per pixel for an image.

## Outlier Detection

|Detector|Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|---|---|---|---|---|
|[Isolation Forest](../methods/iforest.ipynb)|✔|✘|✘|✘|✔|✘|✘|
|[Mahalanobis Distance](../methods/mahalanobis.ipynb)|✔|✘|✘|✘|✔|✔|✘|
|[VAE](../methods/vae.ipynb)|✔|✔|✘|✘|✘|✘|✔|
|[AEGMM](../methods/aegmm.ipynb)|✔|✔|✘|✘|✘|✘|✘|
|[VAEGMM](../methods/vaegmm.ipynb)|✔|✔|✘|✘|✘|✘|✘|

## Adversarial Detection

|Detector|Tabular|Image|Time Series|Text|Categorical Features|Online|Feature Level|
|---|---|---|---|---|
|[Adversarial VAE](../methods/adversarialvae.ipynb)|✔|✔|✘|✘|✘|✘|✘|