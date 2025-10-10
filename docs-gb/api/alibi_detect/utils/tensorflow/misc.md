# `alibi_detect.utils.tensorflow.misc`
## Functions
### `clone_model`

```python
clone_model(model: keras.src.models.model.Model) -> keras.src.models.model.Model
```

Clone a sequential, functional or subclassed tf.keras.Model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `keras.src.models.model.Model` |  |  |

**Returns**
- Type: `keras.src.models.model.Model`

### `quantile`

```python
quantile(sample: tensorflow.python.framework.tensor.Tensor, p: float, type: int = 7, sorted: bool = False) -> float
```

Estimate a desired quantile of a univariate distribution from a vector of samples

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `sample` | `tensorflow.python.framework.tensor.Tensor` |  | A 1D vector of values |
| `p` | `float` |  | The desired quantile in (0,1) |
| `type` | `int` | `7` | The method for computing the quantile. See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample |
| `sorted` | `bool` | `False` | Whether or not the vector is already sorted into ascending order |

**Returns**
- Type: `float`

### `subset_matrix`

```python
subset_matrix(mat: tensorflow.python.framework.tensor.Tensor, inds_0: tensorflow.python.framework.tensor.Tensor, inds_1: tensorflow.python.framework.tensor.Tensor) -> tensorflow.python.framework.tensor.Tensor
```

Take a matrix and return the submatrix correspond to provided row and column indices

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `mat` | `tensorflow.python.framework.tensor.Tensor` |  | A 2D matrix |
| `inds_0` | `tensorflow.python.framework.tensor.Tensor` |  | A vector of row indices |
| `inds_1` | `tensorflow.python.framework.tensor.Tensor` |  | A vector of column indices |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `zero_diag`

```python
zero_diag(mat: tensorflow.python.framework.tensor.Tensor) -> tensorflow.python.framework.tensor.Tensor
```

Set the diagonal of a matrix to 0

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `mat` | `tensorflow.python.framework.tensor.Tensor` |  | A 2D square matrix |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`
