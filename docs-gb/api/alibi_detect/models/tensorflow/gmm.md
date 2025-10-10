# `alibi_detect.models.tensorflow.gmm`
## Functions
### `gmm_energy`

```python
gmm_energy(z: tensorflow.python.framework.tensor.Tensor, phi: tensorflow.python.framework.tensor.Tensor, mu: tensorflow.python.framework.tensor.Tensor, cov: tensorflow.python.framework.tensor.Tensor, L: tensorflow.python.framework.tensor.Tensor, log_det_cov: tensorflow.python.framework.tensor.Tensor, return_mean: bool = True) -> Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]
```

Compute sample energy from Gaussian Mixture Model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `z` | `tensorflow.python.framework.tensor.Tensor` |  | Observations. |
| `phi` | `tensorflow.python.framework.tensor.Tensor` |  | Mixture component distribution weights. |
| `mu` | `tensorflow.python.framework.tensor.Tensor` |  | Mixture means. |
| `cov` | `tensorflow.python.framework.tensor.Tensor` |  | Mixture covariance. |
| `L` | `tensorflow.python.framework.tensor.Tensor` |  | Cholesky decomposition of `cov`. |
| `log_det_cov` | `tensorflow.python.framework.tensor.Tensor` |  | Log of the determinant of `cov`. |
| `return_mean` | `bool` | `True` | Take mean across all sample energies in a batch. |

**Returns**
- Type: `Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]`

### `gmm_params`

```python
gmm_params(z: tensorflow.python.framework.tensor.Tensor, gamma: tensorflow.python.framework.tensor.Tensor) -> Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]
```

Compute parameters of Gaussian Mixture Model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `z` | `tensorflow.python.framework.tensor.Tensor` |  | Observations. |
| `gamma` | `tensorflow.python.framework.tensor.Tensor` |  | Mixture probabilities to derive mixture distribution weights from. |

**Returns**
- Type: `Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]`
