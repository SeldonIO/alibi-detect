# Roadmap

Alibi-detect aims to be the go-to library for outlier, adversarial and concept drift detection in Python. 

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
tabular data, images and time series as well as an offline adversarial detector for 
tabular data and images. Current drift detection capabilities cover tabular data, text and images.

The **near term** focus will be on extending drift, outlier and adversarial detectors for 
text data and adding outlier detectors for mixed data types.

In the **medium term**, we intend to leverage labels in a semi-supervised setting for the
detectors and incorporate drift detection for time series.