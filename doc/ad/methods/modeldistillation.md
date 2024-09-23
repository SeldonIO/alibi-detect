---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.ad.model_distillation.rst)

# Model distillation

## Overview

[Model distillation](https://arxiv.org/abs/1503.02531) is a technique that is used to transfer knowledge from a large network to a smaller network. Typically, it consists of training a second model with a simplified architecture on soft targets (the output distributions or the logits) obtained from the original model. 

Here, we apply model distillation to obtain harmfulness scores, by comparing the output distributions of the original model with the output distributions 
of the distilled model, in order to detect adversarial data, malicious data drift or data corruption.
We use the following definition of harmful and harmless data points:

* Harmful data points are defined as inputs for which the model's predictions on the uncorrupted data are correct while the model's predictions on the corrupted data are wrong.

* Harmless data points are defined as inputs for which the model's predictions on the uncorrupted data are correct and the model's predictions on the corrupted data remain correct.

Analogously to the [adversarial AE detector](https://arxiv.org/abs/2002.09364), which is also part of the library, the model distillation detector picks up drift that reduces the performance of the classification model. 

The detector can be used as follows:

* Given an input $x,$ an adversarial score $S(x)$ is computed. $S(x)$ equals the value loss function employed for distillation calculated between the original model's output and the distilled model's output on $x$.

* If $S(x)$ is above a threshold (explicitly defined or inferred from training data), the instance is flagged as adversarial.

## Usage

### Initialize

Parameters:

* `threshold`: threshold value above which the instance is flagged as an adversarial instance.

* `distilled_model`: `tf.keras.Sequential` instance containing the model used for distillation. Example:

```python
distilled_model = tf.keras.Sequential(
    [
        tf.keras.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax)
    ]
)
```

* `model`: the classifier as a `tf.keras.Model`. Example:

```python
inputs = tf.keras.Input(shape=(input_dim,))
hidden = tf.keras.layers.Dense(hidden_dim)(inputs)
outputs = tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax)(hidden)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

* `loss_type`: type of loss used for distillation. Supported losses: 'kld', 'xent'.

* `temperature`: Temperature used for model prediction scaling. Temperature <1 sharpens the prediction probability distribution which can be beneficial for prediction distributions with high entropy.

* `data_type`: can specify data type added to metadata. E.g. *'tabular'* or *'image'*.

Initialized detector example:

```python
from alibi_detect.ad import ModelDistillation

ad = ModelDistillation(
    distilled_model=distilled_model,
    model=model,
    temperature=0.5
)
```

### Fit

We then need to train the detector. The following parameters can be specified:

* `X`: training batch as a numpy array.

* `loss_fn`: loss function used for training. Defaults to the custom model distillation loss.

* `optimizer`: optimizer used for training. Defaults to [Adam](https://arxiv.org/abs/1412.6980) with learning rate 1e-3.

* `epochs`: number of training epochs.

* `batch_size`: batch size used during training.

* `verbose`: boolean whether to print training progress.

* `log_metric`: additional metrics whose progress will be displayed if verbose equals True.

* `preprocess_fn`: optional data preprocessing function applied per batch during training.


```python
ad.fit(X_train, epochs=50)
```

The threshold for the adversarial / harmfulness score can be set via ```infer_threshold```. We need to pass a batch of instances $X$ and specify what percentage of those we consider to be normal via `threshold_perc`. Even if we only have normal instances in the batch, it might be best to set the threshold value a bit lower (e.g. $95$%) since  the model could have misclassified training instances.

```python
ad.infer_threshold(X_train, threshold_perc=95, batch_size=64)
```

### Detect

We detect adversarial / harmful instances by simply calling `predict` on a batch of instances `X`. We can also return the instance level score by setting `return_instance_score` to True.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_adversarial`: boolean whether instances are above the threshold and therefore adversarial instances. The array is of shape *(batch size,)*.

* `instance_score`: contains instance level scores if `return_instance_score` equals True.


```python
preds_detect = ad.predict(X, batch_size=64, return_instance_score=True)
```

## Examples

### Image

[Harmful drift detection through model distillation on CIFAR10](../../examples/cd_distillation_cifar10.ipynb)

