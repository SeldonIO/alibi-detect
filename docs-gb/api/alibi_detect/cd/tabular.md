# `alibi_detect.cd.tabular`
## `TabularDrift`

_Inherits from:_ `BaseUnivariateDrift`, `BaseDetector`, `ABC`, `DriftConfigMixin`

### Constructor

```python
TabularDrift(self, x_ref: Union[numpy.ndarray, list], p_val: float = 0.05, categories_per_feature: Dict[int, Optional[int]] = None, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, correction: str = 'bonferroni', alternative: str = 'two-sided', n_features: Optional[int] = None, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for significance of the K-S and Chi2 test for each feature. If the FDR correction method is used, this corresponds to the acceptable q-value. |
| `categories_per_feature` | `Dict[int, Optional[int]]` | `None` | Dictionary with as keys the column indices of the categorical features and optionally as values the number of possible categorical values for that feature or a list with the possible values. If you know which features are categorical and simply want to infer the possible values of the categorical feature from the reference data you can pass a Dict[int, NoneType] such as {0: None, 3: None} if features 0 and 3 are categorical. If you also know how many categories are present for a given feature you could pass this in the `categories_per_feature` dict in the Dict[int, int] format, e.g. *{0: 3, 3: 2}*. If you pass N categories this will assume the possible values for the feature are [0, ..., N-1]. You can also explicitly pass the possible categories in the Dict[int, List[int]] format, e.g. {0: [0, 1, 2], 3: [0, 55]}. Note that the categories can be arbitrary int values. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. Typically a dimensionality reduction technique. |
| `correction` | `str` | `'bonferroni'` | Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate). |
| `alternative` | `str` | `'two-sided'` | Defines the alternative hypothesis for the K-S tests. Options are 'two-sided', 'less' or 'greater'. |
| `n_features` | `Optional[int]` | `None` | Number of features used in the combined K-S/Chi-Squared tests. No need to pass it if no preprocessing takes place. In case of a preprocessing step, this can also be inferred automatically but could be more expensive to compute. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `feature_score`

```python
feature_score(x_ref: numpy.ndarray, x: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Compute K-S or Chi-Squared test statistics and p-values per feature.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `numpy.ndarray` |  | Reference instances to compare distribution with. |
| `x` | `numpy.ndarray` |  | Batch of instances. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`
