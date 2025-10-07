# `alibi_detect.utils.tensorflow.perturbation`
## Functions
### `mutate_categorical`

```python
mutate_categorical(X: numpy.ndarray, rate: Optional[float] = None, seed: int = 0, feature_range: tuple = (0, 255)) -> tensorflow.python.framework.tensor.Tensor
```

Randomly change integer feature values to values within a set range

with a specified permutation rate.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of data to be perturbed. |
| `rate` | `Optional[float]` | `None` | Permutation rate (between 0 and 1). |
| `seed` | `int` | `0` | Random seed. |
| `feature_range` | `tuple` | `(0, 255)` | Min and max range for perturbed features. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`
