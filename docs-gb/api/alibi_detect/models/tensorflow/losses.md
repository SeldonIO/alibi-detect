# alibi\_detect.models.tensorflow.losses

## Functions

### `elbo`

```python
elbo(y_true: tensorflow.python.framework.tensor.Tensor, y_pred: tensorflow.python.framework.tensor.Tensor, cov_full: Optional[tensorflow.python.framework.tensor.Tensor] = None, cov_diag: Optional[tensorflow.python.framework.tensor.Tensor] = None, sim: Optional[float] = None) -> tensorflow.python.framework.tensor.Tensor
```

Compute ELBO loss. The covariance matrix can be specified by passing the full covariance matrix, the matrix

diagonal, or a scale identity multiplier. Only one of these should be specified. If none are specified, the identity matrix is used.

## Example

> > > import tensorflow as tf from alibi\_detect.models.tensorflow.losses import elbo y\_true = tf.constant(\[\[0.0, 1.0], \[1.0, 0.0]]) y\_pred = tf.constant(\[\[0.1, 0.9], \[0.8, 0.2]])
> > >
> > > ## Specifying scale identity multiplier
> > >
> > > elbo(y\_true, y\_pred, sim=1.0)
> > >
> > > ## Specifying covariance matrix diagonal
> > >
> > > elbo(y\_true, y\_pred, cov\_diag=tf.ones(2))
> > >
> > > ## Specifying full covariance matrix
> > >
> > > elbo(y\_true, y\_pred, cov\_full=tf.eye(2))

| Name       | Type                                                  | Default | Description                               |
| ---------- | ----------------------------------------------------- | ------- | ----------------------------------------- |
| `y_true`   | `tensorflow.python.framework.tensor.Tensor`           |         | Labels.                                   |
| `y_pred`   | `tensorflow.python.framework.tensor.Tensor`           |         | Predictions.                              |
| `cov_full` | `Optional[tensorflow.python.framework.tensor.Tensor]` | `None`  | Full covariance matrix.                   |
| `cov_diag` | `Optional[tensorflow.python.framework.tensor.Tensor]` | `None`  | Diagonal (variance) of covariance matrix. |
| `sim`      | `Optional[float]`                                     | `None`  | Scale identity multiplier.                |

**Returns**

* Type: `tensorflow.python.framework.tensor.Tensor`

### `loss_adv_ae`

```python
loss_adv_ae(x_true: tensorflow.python.framework.tensor.Tensor, x_pred: tensorflow.python.framework.tensor.Tensor, model: Optional[keras.src.models.model.Model] = None, model_hl: Optional[list] = None, w_model: float = 1.0, w_recon: float = 0.0, w_model_hl: Optional[list] = None, temperature: float = 1.0) -> tensorflow.python.framework.tensor.Tensor
```

Loss function used for AdversarialAE.

| Name          | Type                                        | Default | Description                                                                                                     |
| ------------- | ------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------- |
| `x_true`      | `tensorflow.python.framework.tensor.Tensor` |         | Batch of instances.                                                                                             |
| `x_pred`      | `tensorflow.python.framework.tensor.Tensor` |         | Batch of reconstructed instances by the autoencoder.                                                            |
| `model`       | `Optional[keras.src.models.model.Model]`    | `None`  | A trained tf.keras model with frozen layers (layers.trainable = False).                                         |
| `model_hl`    | `Optional[list]`                            | `None`  | List with tf.keras models used to extract feature maps and make predictions on hidden layers.                   |
| `w_model`     | `float`                                     | `1.0`   | Weight on model prediction loss term.                                                                           |
| `w_recon`     | `float`                                     | `0.0`   | Weight on MSE reconstruction error loss term.                                                                   |
| `w_model_hl`  | `Optional[list]`                            | `None`  | Weights assigned to the loss of each model in model\_hl.                                                        |
| `temperature` | `float`                                     | `1.0`   | Temperature used for model prediction scaling. Temperature <1 sharpens the prediction probability distribution. |

**Returns**

* Type: `tensorflow.python.framework.tensor.Tensor`

### `loss_aegmm`

```python
loss_aegmm(x_true: tensorflow.python.framework.tensor.Tensor, x_pred: tensorflow.python.framework.tensor.Tensor, z: tensorflow.python.framework.tensor.Tensor, gamma: tensorflow.python.framework.tensor.Tensor, w_energy: float = 0.1, w_cov_diag: float = 0.005) -> tensorflow.python.framework.tensor.Tensor
```

Loss function used for OutlierAEGMM.

| Name         | Type                                        | Default | Description                                          |
| ------------ | ------------------------------------------- | ------- | ---------------------------------------------------- |
| `x_true`     | `tensorflow.python.framework.tensor.Tensor` |         | Batch of instances.                                  |
| `x_pred`     | `tensorflow.python.framework.tensor.Tensor` |         | Batch of reconstructed instances by the autoencoder. |
| `z`          | `tensorflow.python.framework.tensor.Tensor` |         | Latent space values.                                 |
| `gamma`      | `tensorflow.python.framework.tensor.Tensor` |         | Membership prediction for mixture model components.  |
| `w_energy`   | `float`                                     | `0.1`   | Weight on sample energy loss term.                   |
| `w_cov_diag` | `float`                                     | `0.005` | Weight on covariance regularizing loss term.         |

**Returns**

* Type: `tensorflow.python.framework.tensor.Tensor`

### `loss_distillation`

```python
loss_distillation(x_true: tensorflow.python.framework.tensor.Tensor, y_pred: tensorflow.python.framework.tensor.Tensor, model: Optional[keras.src.models.model.Model] = None, loss_type: str = 'kld', temperature: float = 1.0) -> tensorflow.python.framework.tensor.Tensor
```

Loss function used for Model Distillation.

| Name          | Type                                        | Default | Description                                                                                                     |
| ------------- | ------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------- |
| `x_true`      | `tensorflow.python.framework.tensor.Tensor` |         | Batch of data points.                                                                                           |
| `y_pred`      | `tensorflow.python.framework.tensor.Tensor` |         | Batch of prediction from the distilled model.                                                                   |
| `model`       | `Optional[keras.src.models.model.Model]`    | `None`  | tf.keras model.                                                                                                 |
| `loss_type`   | `str`                                       | `'kld'` | Type of loss for distillation. Supported 'kld', 'xent.                                                          |
| `temperature` | `float`                                     | `1.0`   | Temperature used for model prediction scaling. Temperature <1 sharpens the prediction probability distribution. |

**Returns**

* Type: `tensorflow.python.framework.tensor.Tensor`

### `loss_vaegmm`

```python
loss_vaegmm(x_true: tensorflow.python.framework.tensor.Tensor, x_pred: tensorflow.python.framework.tensor.Tensor, z: tensorflow.python.framework.tensor.Tensor, gamma: tensorflow.python.framework.tensor.Tensor, w_recon: float = 1e-07, w_energy: float = 0.1, w_cov_diag: float = 0.005, cov_full: Optional[tensorflow.python.framework.tensor.Tensor] = None, cov_diag: Optional[tensorflow.python.framework.tensor.Tensor] = None, sim: float = 0.05) -> tensorflow.python.framework.tensor.Tensor
```

Loss function used for OutlierVAEGMM.

| Name         | Type                                                  | Default | Description                                                      |
| ------------ | ----------------------------------------------------- | ------- | ---------------------------------------------------------------- |
| `x_true`     | `tensorflow.python.framework.tensor.Tensor`           |         | Batch of instances.                                              |
| `x_pred`     | `tensorflow.python.framework.tensor.Tensor`           |         | Batch of reconstructed instances by the variational autoencoder. |
| `z`          | `tensorflow.python.framework.tensor.Tensor`           |         | Latent space values.                                             |
| `gamma`      | `tensorflow.python.framework.tensor.Tensor`           |         | Membership prediction for mixture model components.              |
| `w_recon`    | `float`                                               | `1e-07` | Weight on elbo loss term.                                        |
| `w_energy`   | `float`                                               | `0.1`   | Weight on sample energy loss term.                               |
| `w_cov_diag` | `float`                                               | `0.005` | Weight on covariance regularizing loss term.                     |
| `cov_full`   | `Optional[tensorflow.python.framework.tensor.Tensor]` | `None`  | Full covariance matrix.                                          |
| `cov_diag`   | `Optional[tensorflow.python.framework.tensor.Tensor]` | `None`  | Diagonal (variance) of covariance matrix.                        |
| `sim`        | `float`                                               | `0.05`  | Scale identity multiplier.                                       |

**Returns**

* Type: `tensorflow.python.framework.tensor.Tensor`
