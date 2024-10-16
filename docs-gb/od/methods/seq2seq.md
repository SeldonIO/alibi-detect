---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.od.seq2seq.rst)

# Sequence-to-Sequence (Seq2Seq)

## Overview

The [Sequence-to-Sequence](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) (Seq2Seq) outlier detector consists of 2 main building blocks: an encoder and a decoder. The encoder consists of a [Bidirectional](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks) [LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) which processes the input sequence and initializes the decoder. The LSTM decoder then makes sequential predictions for the output sequence. In our case, the decoder aims to reconstruct the input sequence. If the input data cannot be reconstructed well, the reconstruction error is high and the data can be flagged as an outlier. The reconstruction error is measured as the mean squared error (MSE) between the input and the reconstructed instance. 

Since even for normal data the reconstruction error can be state-dependent, we add an outlier threshold estimator network to the Seq2Seq model. This network takes in the hidden state of the decoder at each timestep and predicts the estimated reconstruction error for normal data. As a result, the outlier threshold is not static and becomes a function of the model state. This is similar to [Park et al. (2017)](https://arxiv.org/pdf/1711.00614.pdf), but while they train the threshold estimator separately from the Seq2Seq model with a Support-Vector Regressor, we train a neural net regression network end-to-end with the Seq2Seq model.

The detector is first trained on a batch of unlabeled, but normal (*inlier*) data. Unsupervised training is desireable since labeled data is often scarce. The Seq2Seq outlier detector is suitable for both **univariate and multivariate time series**.

## Usage

### Initialize

Parameters:

* `n_features`: number of features in the time series.

* `seq_len`: sequence length fed into the Seq2Seq model.

* `threshold`: threshold used for outlier detection. Can be a float or feature-wise array.

* `seq2seq`: optionally pass an already defined or pretrained Seq2Seq model to the outlier detector as a `tf.keras.Model`.

* `threshold_net`: optionally pass the layers for the threshold estimation network wrapped in a `tf.keras.Sequential` instance. Example:

```python
threshold_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(seq_len, latent_dim)),
        Dense(64, activation=tf.nn.relu),
        Dense(64, activation=tf.nn.relu),
    ])
```

* `latent_dim`: latent dimension of the encoder and decoder.

* `output_activation`: activation used in the Dense output layer of the decoder.

* `beta`: weight on the threshold estimation mean-squared error (MSE) loss term.

Initialized outlier detector example:

```python
from alibi_detect.od import OutlierSeq2Seq

n_features = 2
seq_len = 50

od = OutlierSeq2Seq(n_features,
                    seq_len,
                    threshold=None,
                    threshold_net=threshold_net,
                    latent_dim=100)
```

### Fit

We then need to train the outlier detector. The following parameters can be specified:

* `X`: univariate or multivariate time series array with preferably normal data used for training. Shape equals *(batch, n_features)* or *(batch, seq_len, n_features)*.

* `loss_fn`: loss function used for training. Defaults to the MSE loss.

* `optimizer`: optimizer used for training. Defaults to [Adam](https://arxiv.org/abs/1412.6980) with learning rate 1e-3.

* `epochs`: number of training epochs.

* `batch_size`: batch size used during training.

* `verbose`: boolean whether to print training progress.

* `log_metric`: additional metrics whose progress will be displayed if verbose equals True.


```python
od.fit(X_train, epochs=20)
```

It is often hard to find a good threshold value. If we have a batch of normal and outlier data and we know approximately the percentage of normal data in the batch, we can infer a suitable threshold. We can either set the threshold over both features combined or determine a feature-wise threshold. Here we opt for the feature-wise threshold. This is for instance useful when different features have different variance or sensitivity to outliers. The snippet assumes there are about 5% outliers in the first feature and 10% in the second:

```python
od.infer_threshold(X, threshold_perc=np.array([95, 90]))
```

### Detect

We detect outliers by simply calling `predict` on a batch of instances `X`. Detection can be customized via the following parameters:

* `outlier_type`: either *'instance'* or *'feature'*. If the outlier type equals *'instance'*, the outlier score at the instance level will be used to classify the instance as an outlier or not. If *'feature'* is selected, outlier detection happens at the feature level. It is **important to distinguish 2 use cases**:
  * `X` has shape *(batch, n_features)*:
    * There are *batch* instances with *n_features* features per instance.
  
  * `X` has shape *(batch, seq_len, n_features)*
    * Now there are *batch* instances with *seq_len x n_features* features per instance.

* `outlier_perc`: percentage of the sorted (descending) feature level outlier scores. We might for instance want to flag a multivariate time series as an outlier at a specific timestamp if at least 75% of the feature values are on average above the threshold. In this case, we set `outlier_perc` to 75. The default value is 100 (using all the features).

* `return_feature_score`: boolean whether to return the feature level outlier scores.

* `return_instance_score`: boolean whether to return the instance level outlier scores.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_outlier`: boolean whether instances or features are above the threshold and therefore outliers. If `outlier_type` equals *'instance'*, then the array is of shape *(batch,)*. If it equals *'feature'*, then the array is of shape *(batch, seq_len, n_features)* or *(batch, n_features)*, depending on the shape of `X`.

* `feature_score`: contains feature level scores if `return_feature_score` equals True.

* `instance_score`: contains instance level scores if `return_instance_score` equals True.


```python
preds = od.predict(X,
                   outlier_type='instance',
                   outlier_perc=100,
                   return_feature_score=True,
                   return_instance_score=True)
```

## Examples

[Time series outlier detection with Seq2Seq models on synthetic data](../../examples/od_seq2seq_synth.ipynb)

[Seq2Seq time series outlier detection on ECG data](../../examples/od_seq2seq_ecg.ipynb)

