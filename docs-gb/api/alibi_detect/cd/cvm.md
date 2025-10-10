# `alibi_detect.cd.cvm`
## `CVMDrift`

_Inherits from:_ `BaseUnivariateDrift`, `BaseDetector`, `ABC`, `DriftConfigMixin`

### Constructor

```python
CVMDrift(self, x_ref: Union[numpy.ndarray, list], p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, correction: str = 'bonferroni', n_features: Optional[int] = None, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for significance of the CVM test. If the FDR correction method is used, this corresponds to the acceptable q-value. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `correction` | `str` | `'bonferroni'` | Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate). |
| `n_features` | `Optional[int]` | `None` | Number of features used in the CVM test. No need to pass it if no preprocessing takes place. In case of a preprocessing step, this can also be inferred automatically but could be more expensive to compute. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `feature_score`

```python
feature_score(x_ref: numpy.ndarray, x: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Performs the two-sample Cramer-von Mises test(s), computing the p-value and test statistic per feature.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `numpy.ndarray` |  | Reference instances to compare distribution with. |
| `x` | `numpy.ndarray` |  | Batch of instances. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`
