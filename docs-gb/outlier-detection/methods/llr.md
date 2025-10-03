---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.od.llr.rst)

# Likelihood Ratios for Outlier Detection

## Overview

The outlier detector described by [Ren et al. (2019)](https://arxiv.org/abs/1906.02845) in [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845) uses the likelihood ratio (LLR) between 2 generative models as the outlier score. One model is trained on the original data while the other is trained on a perturbed version of the dataset. This is based on the observation that the log likelihood for an instance under a generative model can be heavily affected by population level background statistics. The second generative model is therefore trained to capture the background statistics still present in the perturbed data while the semantic features have been erased by the perturbations.

The perturbations are added using an independent and identical Bernoulli distribution with rate $\mu$ which substitutes a feature with one of the other possible feature values with equal probability. For images, this means for instance changing a pixel with a different pixel value randomly sampled within the $0$ to $255$ pixel range. The package also contains a [PixelCNN++](https://arxiv.org/abs/1701.05517) implementation adapted from the official TensorFlow Probability [version](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PixelCNN), and available as a standalone model in `alibi_detect.models.tensorflow.pixelcnn`.

## Usage

### Initialize

Parameters:

* `threshold`: outlier threshold value used for the negative likelihood ratio. Scores above the threshold are flagged as outliers.

* `model`: a generative model, either as a `tf.keras.Model`, TensorFlow Probability distribution or built-in PixelCNN++ model.

* `model_background`: optional separate model fit on the perturbed background data. If this is not specified, a copy of `model` will be used.

* `log_prob`: if the model does not have a `log_prob` function like e.g. a TensorFlow Probability distribution, a function needs to be passed that evaluates the log likelihood.

* `sequential`: flag whether the data is sequential or not. Used to create targets during training. Defaults to *False*.

* `data_type`: can specify data type added to metadata. E.g. *'tabular'* or *'image'*.

Initialized outlier detector example:

```python
from alibi_detect.od import LLR
from alibi_detect.models.tensorflow import PixelCNN

image_shape = (28, 28, 1)
model = PixelCNN(image_shape)
od = LLR(threshold=-100, model=model)
```

### Fit

We then need to train the 2 generative models in sequence. The following parameters can be specified:

* `X`: training batch as a numpy array of preferably normal data.

* `mutate_fn`: function used to create the perturbations. Defaults to an independent and identical Bernoulli distribution with rate $\mu$ 

* `mutate_fn_kwargs`: kwargs for `mutate_fn`. For the default function, the mutation rate and feature range needs to be specified, e.g. *dict(rate=.2, feature_range=(0,255))*.

* `loss_fn`: loss function used for the generative models.

* `loss_fn_kwargs`: kwargs for the loss function.

* `optimizer`: optimizer used for training. Defaults to [Adam](https://arxiv.org/abs/1412.6980) with learning rate 1e-3.

* `epochs`: number of training epochs.

* `batch_size`: batch size used during training.

* `log_metric`: additional metrics whose progress will be displayed if verbose equals True.

```python
od.fit(X_train, epochs=10, batch_size=32)
```

It is often hard to find a good threshold value. If we have a batch of normal and outlier data and we know approximately the percentage of normal data in the batch, we can infer a suitable threshold:

```python
od.infer_threshold(X, threshold_perc=95, batch_size=32)
```

### Detect

We detect outliers by simply calling `predict` on a batch of instances `X`. Detection can be customized via the following parameters:

* `outlier_type`: either *'instance'* or *'feature'*. If the outlier type equals *'instance'*, the outlier score at the instance level will be used to classify the instance as an outlier or not. If *'feature'* is selected, outlier detection happens at the feature level (e.g. by pixel in images).

* `batch_size`: batch size used for model prediction calls.

* `return_feature_score`: boolean whether to return the feature level outlier scores.

* `return_instance_score`: boolean whether to return the instance level outlier scores.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_outlier`: boolean whether instances or features are above the threshold and therefore outliers. If `outlier_type` equals *'instance'*, then the array is of shape *(batch size,)*. If it equals *'feature'*, then the array is of shape *(batch size, instance shape)*.

* `feature_score`: contains feature level scores if `return_feature_score` equals True.

* `instance_score`: contains instance level scores if `return_instance_score` equals True.


```python
preds = od.predict(X, outlier_type='instance', batch_size=32)
```

## Examples

### Image

[Likelihood Ratio Outlier Detection with PixelCNN++](../../examples/od_llr_mnist.ipynb)

### Sequential Data

[Likelihood Ratio Outlier Detection on Genomic Sequences](../../examples/od_llr_genome.ipynb)

