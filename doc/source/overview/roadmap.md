# Roadmap

Alibi Detect aims to be the go-to library for **outlier**, **adversarial** and **drift** detection in Python using 
both the **TensorFlow** and **PyTorch** backends.

This means that the algorithms in the library need to handle:
* **Online** detection with often stateful detectors.
* **Offline** detection, where the detector is trained on a batch of unsupervised or semi-supervised data. This assumption resembles a lot of real-world settings where labels are hard to come by.

The algorithms will cover the following data types:
* **Tabular**, including both numerical and categorical data.
* **Images**
* **Time series**, both univariate and multivariate.
* **Text**

It will also be possible to combine different algorithms in ensemble detectors.

The library **currently** covers both online and offline outlier detection algorithms for 
tabular data, images and time series as well as offline adversarial detectors for 
tabular data and images. Current drift detection capabilities cover mixed type tabular data, text and images.

The **near term** focus will be on adding online and text drift detectors, extending the PyTorch support, and adding outlier detectors for text and mixed data types.

In the **medium term**, we intend to leverage labels in a semi-supervised setting for the
detectors and incorporate drift detection for time series.