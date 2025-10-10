# `alibi_detect.utils.discretizer`
## `Discretizer`

### Constructor

```python
Discretizer(self, data: numpy.ndarray, categorical_features: List[int], feature_names: List[str], percentiles: List[int] = [25, 50, 75]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | Data to discretize |
| `categorical_features` | `List[int]` |  | List of indices corresponding to the categorical columns. These features will not be discretized. The other features will be considered continuous and therefore discretized. |
| `feature_names` | `List[str]` |  | List with feature names |
| `percentiles` | `List[int]` | `[25, 50, 75]` | Percentiles used for discretization |

### Methods

#### `bins`

```python
bins(data: numpy.ndarray) -> List[numpy.ndarray]
```

Parameters

----------
data
    Data to discretize

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | Data to discretize |

**Returns**
- Type: `List[numpy.ndarray]`

#### `discretize`

```python
discretize(data: numpy.ndarray) -> numpy.ndarray
```

Parameters

----------
data
    Data to discretize

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | Data to discretize |

**Returns**
- Type: `numpy.ndarray`
