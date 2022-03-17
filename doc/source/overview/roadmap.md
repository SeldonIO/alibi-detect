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
* **Graphs**

It will also be possible to combine different algorithms in ensemble detectors.

The library **currently** covers both online and offline **outlier** detection algorithms for 
tabular data, images and time series as well as offline **adversarial** detectors for 
tabular data and images. Current **drift** detection capabilities cover almost any data modality such as mixed type tabular data, 
text, images or graphs, both in the online and offline setting. Furthermore, Alibi Detect provides supervised drift and context-aware drift detectors.

The **near term** focus will be on extending save/load functionality for PyTorch detectors, and adding outlier detectors for text and mixed data types.
