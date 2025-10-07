# `alibi_detect.utils.misc`
## Functions
### `quantile`

```python
quantile(sample: numpy.ndarray, p: float, type: int = 7, sorted: bool = False, interpolate: bool = True) -> float
```

Estimate a desired quantile of a univariate distribution from a vector of samples

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `sample` | `numpy.ndarray` |  | A 1D vector of values |
| `p` | `float` |  | The desired quantile in (0,1) |
| `type` | `int` | `7` | The method for computing the quantile. See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample |
| `sorted` | `bool` | `False` | Whether or not the vector is already sorted into ascending order |
| `interpolate` | `bool` | `True` | Whether to interpolate the desired quantile. |

**Returns**
- Type: `float`
