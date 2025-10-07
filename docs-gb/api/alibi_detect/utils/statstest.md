# `alibi_detect.utils.statstest`
## Functions
### `fdr`

```python
fdr(p_val: numpy.ndarray, q_val: float) -> Tuple[int, Union[float, numpy.ndarray]]
```

Checks the significance of univariate tests on each variable between 2 samples of

multivariate data via the False Discovery Rate (FDR) correction of the p-values.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `p_val` | `numpy.ndarray` |  | p-values for each univariate test. |
| `q_val` | `float` |  | Acceptable q-value threshold. |

**Returns**
- Type: `Tuple[int, Union[float, numpy.ndarray]]`

### `permutation_test`

```python
permutation_test(x: numpy.ndarray, y: numpy.ndarray, metric: Callable, n_permutations: int = 100, kwargs) -> Tuple[float, float, numpy.ndarray]
```

Apply a permutation test to samples x and y.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Batch of instances of shape [Nx, features]. |
| `y` | `numpy.ndarray` |  | Batch of instances of shape [Ny, features]. |
| `metric` | `Callable` |  | Distance metric used for the test. Defaults to Maximum Mean Discrepancy. |
| `n_permutations` | `int` | `100` | Number of permutations used in the test. |
| `kwargs` |  |  | Kwargs for the metric. For the default this includes for instance the kernel used. |

**Returns**
- Type: `Tuple[float, float, numpy.ndarray]`
