# `alibi_detect.utils.data`
## `Bunch`

_Inherits from:_ `dict`

Container object for internal datasets

Dictionary-like object that exposes its keys as attributes.

### Constructor

```python
Bunch(self, **kwargs)
```

## Functions
### `create_outlier_batch`

```python
create_outlier_batch(data: numpy.ndarray, target: numpy.ndarray, n_samples: int, perc_outlier: int) -> Union[alibi_detect.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]
```

Create a batch with a defined percentage of outliers.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  |  |
| `target` | `numpy.ndarray` |  |  |
| `n_samples` | `int` |  |  |
| `perc_outlier` | `int` |  |  |

**Returns**
- Type: `Union[alibi_detect.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]`

### `sample_df`

```python
sample_df(df: pandas.core.frame.DataFrame, n: int)
```

Sample n instances from the dataframe df.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `df` | `pandas.core.frame.DataFrame` |  |  |
| `n` | `int` |  |  |
