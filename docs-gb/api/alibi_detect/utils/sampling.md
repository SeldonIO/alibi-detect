# `alibi_detect.utils.sampling`
## Functions
### `reservoir_sampling`

```python
reservoir_sampling(X_ref: numpy.ndarray, X: numpy.ndarray, reservoir_size: int, n: int) -> numpy.ndarray
```

Apply reservoir sampling.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ref` | `numpy.ndarray` |  | Current instances in reservoir. |
| `X` | `numpy.ndarray` |  | Data to update reservoir with. |
| `reservoir_size` | `int` |  | Size of reservoir. |
| `n` | `int` |  | Number of total instances that have passed so far. |

**Returns**
- Type: `numpy.ndarray`
