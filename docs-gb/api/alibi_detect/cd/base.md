# `alibi_detect.cd.base`
## Constants
### `has_pytorch`
```python
has_pytorch: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `has_tensorflow`
```python
has_tensorflow: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.cd.base (WARNING)>
```
Instances of the Logger class represent a single logging channel. A
"logging channel" indicates an area of an application. Exactly how an
"area" is defined is up to the application developer. Since an
application can have any number of areas, logging channels are identified
by a unique string. Application areas can be nested (e.g. an area
of "input processing" might include sub-areas "read CSV files", "read
XLS files" and "read Gnumeric files"). To cater for this natural nesting,
channel names are organized into a namespace hierarchy where levels are
separated by periods, much like the Java or Python package namespace. So
in the instance given above, channel names might be "input" for the upper
level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
There is no arbitrary limit to the depth of nesting.

## `BaseClassifierDrift`

_Inherits from:_ `BaseDetector`, `ABC`

### Constructor

```python
BaseClassifierDrift(self, x_ref: Union[numpy.ndarray, list], p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, preds_type: str = 'probs', binarize_preds: bool = False, train_size: Optional[float] = 0.75, n_folds: Optional[int] = None, retrain_from_scratch: bool = True, seed: int = 0, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `preds_type` | `str` | `'probs'` | Whether the model outputs probabilities or logits |
| `binarize_preds` | `bool` | `False` | Whether to test for discrepency on soft (e.g. probs/logits) model predictions directly with a K-S test or binarise to 0-1 prediction errors and apply a binomial test. |
| `train_size` | `Optional[float]` | `0.75` | Optional fraction (float between 0 and 1) of the dataset used to train the classifier. The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`. |
| `n_folds` | `Optional[int]` | `None` | Optional number of stratified folds used for training. The model preds are then calculated on all the out-of-fold predictions. This allows to leverage all the reference and test data for drift detection at the expense of longer computation. If both `train_size` and `n_folds` are specified, `n_folds` is prioritized. |
| `retrain_from_scratch` | `bool` | `True` | Whether the classifier should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set. |
| `seed` | `int` | `0` | Optional random seed for fold selection. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `get_splits`

```python
get_splits(x_ref: Union[numpy.ndarray, list], x: Union[numpy.ndarray, list], return_splits: bool = True) -> Union[Tuple[Union[numpy.ndarray, list], numpy.ndarray], Tuple[Union[numpy.ndarray, list], numpy.ndarray, Optional[List[Tuple[numpy.ndarray, numpy.ndarray]]]]]
```

Split reference and test data in train and test folds used by the classifier.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `return_splits` | `bool` | `True` | Whether to return the splits. |

**Returns**
- Type: `Union[Tuple[Union[numpy.ndarray, list], numpy.ndarray], Tuple[Union[numpy.ndarray, list], numpy.ndarray, Optional[List[Tuple[numpy.ndarray, numpy.ndarray]]]]]`

#### `predict`

```python
predict(x: Union[numpy.ndarray, list], return_p_val: bool = True, return_distance: bool = True, return_probs: bool = True, return_model: bool = True) -> Dict[str, Dict[str, Union[str, int, float, Callable]]]
```

Predict whether a batch of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `return_p_val` | `bool` | `True` | Whether to return the p-value of the test. |
| `return_distance` | `bool` | `True` | Whether to return a notion of strength of the drift. K-S test stat if binarize_preds=False, otherwise relative error reduction. |
| `return_probs` | `bool` | `True` | Whether to return the instance level classifier probabilities for the reference and test data (0=reference data, 1=test data). The reference and test instances of the associated probabilities are also returned. |
| `return_model` | `bool` | `True` | Whether to return the updated model trained to discriminate reference and test instances. |

**Returns**
- Type: `Dict[str, Dict[str, Union[str, int, float, Callable]]]`

#### `preprocess`

```python
preprocess(x: Union[numpy.ndarray, list]) -> Tuple[Union[numpy.ndarray, list], Union[numpy.ndarray, list]]
```

Data preprocessing before computing the drift scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[Union[numpy.ndarray, list], Union[numpy.ndarray, list]]`

#### `score`

```python
score(x: Union[numpy.ndarray, list]) -> Tuple[float, float, numpy.ndarray, numpy.ndarray, Union[numpy.ndarray, list], Union[numpy.ndarray, list]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  |  |

**Returns**
- Type: `Tuple[float, float, numpy.ndarray, numpy.ndarray, Union[numpy.ndarray, list], Union[numpy.ndarray, list]]`

#### `test_probs`

```python
test_probs(y_oof: numpy.ndarray, probs_oof: numpy.ndarray, n_ref: int, n_cur: int) -> Tuple[float, float]
```

Perform a statistical test of the probabilities predicted by the model against

what we'd expect under the no-change null.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `y_oof` | `numpy.ndarray` |  | Out of fold targets (0 ref, 1 cur) |
| `probs_oof` | `numpy.ndarray` |  | Probabilities predicted by the model |
| `n_ref` | `int` |  | Size of reference window used in training model |
| `n_cur` | `int` |  | Size of current window used in training model |

**Returns**
- Type: `Tuple[float, float]`

## `BaseContextMMDDrift`

_Inherits from:_ `BaseDetector`, `ABC`

### Constructor

```python
BaseContextMMDDrift(self, x_ref: Union[numpy.ndarray, list], c_ref: numpy.ndarray, p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, x_kernel: Callable = None, c_kernel: Callable = None, n_permutations: int = 1000, prop_c_held: float = 0.25, n_folds: int = 5, batch_size: Optional[int] = 256, input_shape: Optional[tuple] = None, data_type: Optional[str] = None, verbose: bool = False) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `c_ref` | `numpy.ndarray` |  | Context for the reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the permutation test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last N instances seen by the detector. The parameter should be passed as a dictionary *{'last': N}*. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `x_kernel` | `Optional[Callable]` | `None` | Kernel defined on the input data, defaults to Gaussian RBF kernel. |
| `c_kernel` | `Optional[Callable]` | `None` | Kernel defined on the context data, defaults to Gaussian RBF kernel. |
| `n_permutations` | `int` | `1000` | Number of permutations used in the permutation test. |
| `prop_c_held` | `float` | `0.25` | Proportion of contexts held out to condition on. |
| `n_folds` | `int` | `5` | Number of cross-validation folds used when tuning the regularisation parameters. |
| `batch_size` | `Optional[int]` | `256` | If not None, then compute batches of MMDs at a time (rather than all at once). |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |
| `verbose` | `bool` | `False` | Whether or not to print progress during configuration. |

### Methods

#### `predict`

```python
predict(x: Union[numpy.ndarray, list], c: numpy.ndarray, return_p_val: bool = True, return_distance: bool = True, return_coupling: bool = False) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]
```

Predict whether a batch of data has drifted from the reference data, given the provided context.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `c` | `numpy.ndarray` |  | Context associated with batch of instances. |
| `return_p_val` | `bool` | `True` | Whether to return the p-value of the permutation test. |
| `return_distance` | `bool` | `True` | Whether to return the conditional MMD test statistic between the new batch and reference data. |
| `return_coupling` | `bool` | `False` | Whether to return the coupling matrices. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[int, float]]]`

#### `preprocess`

```python
preprocess(x: Union[numpy.ndarray, list]) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Data preprocessing before computing the drift scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `score`

```python
score(x: Union[numpy.ndarray, list], c: numpy.ndarray) -> Tuple[float, float, float, Tuple]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  |  |
| `c` | `numpy.ndarray` |  |  |

**Returns**
- Type: `Tuple[float, float, float, Tuple]`

## `BaseLSDDDrift`

_Inherits from:_ `BaseDetector`, `ABC`

### Constructor

```python
BaseLSDDDrift(self, x_ref: Union[numpy.ndarray, list], p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, sigma: Optional[numpy.ndarray] = None, n_permutations: int = 100, n_kernel_centers: Optional[int] = None, lambda_rd_max: float = 0.2, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the permutation test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `sigma` | `Optional[numpy.ndarray]` | `None` | Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma` is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance between reference samples. |
| `n_permutations` | `int` | `100` | Number of permutations used in the permutation test. |
| `n_kernel_centers` | `Optional[int]` | `None` | The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD. Defaults to 1/20th of the reference data. |
| `lambda_rd_max` | `float` | `0.2` | The maximum relative difference between two estimates of LSDD that the regularization parameter lambda is allowed to cause. Defaults to 0.2 as in the paper. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `predict`

```python
predict(x: Union[numpy.ndarray, list], return_p_val: bool = True, return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]
```

Predict whether a batch of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `return_p_val` | `bool` | `True` | Whether to return the p-value of the permutation test. |
| `return_distance` | `bool` | `True` | Whether to return the LSDD metric between the new batch and reference data. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[int, float]]]`

#### `preprocess`

```python
preprocess(x: Union[numpy.ndarray, list]) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Data preprocessing before computing the drift scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `score`

```python
score(x: Union[numpy.ndarray, list]) -> Tuple[float, float, float]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  |  |

**Returns**
- Type: `Tuple[float, float, float]`

## `BaseLearnedKernelDrift`

_Inherits from:_ `BaseDetector`, `ABC`

### Constructor

```python
BaseLearnedKernelDrift(self, x_ref: Union[numpy.ndarray, list], p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, n_permutations: int = 100, train_size: Optional[float] = 0.75, retrain_from_scratch: bool = True, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `n_permutations` | `int` | `100` | The number of permutations to use in the permutation test once the MMD has been computed. |
| `train_size` | `Optional[float]` | `0.75` | Optional fraction (float between 0 and 1) of the dataset used to train the kernel. The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`. |
| `retrain_from_scratch` | `bool` | `True` | Whether the kernel should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `get_splits`

```python
get_splits(x_ref: Union[numpy.ndarray, list], x: Union[numpy.ndarray, list]) -> Tuple[Tuple[Union[numpy.ndarray, list], Union[numpy.ndarray, list]], Tuple[Union[numpy.ndarray, list], Union[numpy.ndarray, list]]]
```

Split reference and test data into two splits -- one of which to learn test locations

and parameters and one to use for tests.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[Tuple[Union[numpy.ndarray, list], Union[numpy.ndarray, list]], Tuple[Union[numpy.ndarray, list], Union[numpy.ndarray, list]]]`

#### `predict`

```python
predict(x: Union[numpy.ndarray, list], return_p_val: bool = True, return_distance: bool = True, return_kernel: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float, Callable]]]
```

Predict whether a batch of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `return_p_val` | `bool` | `True` | Whether to return the p-value of the permutation test. |
| `return_distance` | `bool` | `True` | Whether to return the MMD metric between the new batch and reference data. |
| `return_kernel` | `bool` | `True` | Whether to return the updated kernel trained to discriminate reference and test instances. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[int, float, Callable]]]`

#### `preprocess`

```python
preprocess(x: Union[numpy.ndarray, list]) -> Tuple[Union[numpy.ndarray, list], Union[numpy.ndarray, list]]
```

Data preprocessing before computing the drift scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[Union[numpy.ndarray, list], Union[numpy.ndarray, list]]`

#### `score`

```python
score(x: Union[numpy.ndarray, list]) -> Tuple[float, float, float]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  |  |

**Returns**
- Type: `Tuple[float, float, float]`

## `BaseMMDDrift`

_Inherits from:_ `BaseDetector`, `ABC`

### Constructor

```python
BaseMMDDrift(self, x_ref: Union[numpy.ndarray, list], p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, sigma: Optional[numpy.ndarray] = None, configure_kernel_from_x_ref: bool = True, n_permutations: int = 100, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the permutation test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `sigma` | `Optional[numpy.ndarray]` | `None` | Optionally set the Gaussian RBF kernel bandwidth. Can also pass multiple bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. |
| `configure_kernel_from_x_ref` | `bool` | `True` | Whether to already configure the kernel bandwidth from the reference data. |
| `n_permutations` | `int` | `100` | Number of permutations used in the permutation test. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `predict`

```python
predict(x: Union[numpy.ndarray, list], return_p_val: bool = True, return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]
```

Predict whether a batch of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `return_p_val` | `bool` | `True` | Whether to return the p-value of the permutation test. |
| `return_distance` | `bool` | `True` | Whether to return the MMD metric between the new batch and reference data. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[int, float]]]`

#### `preprocess`

```python
preprocess(x: Union[numpy.ndarray, list]) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Data preprocessing before computing the drift scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `score`

```python
score(x: Union[numpy.ndarray, list]) -> Tuple[float, float, float]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  |  |

**Returns**
- Type: `Tuple[float, float, float]`

## `BaseUnivariateDrift`

_Inherits from:_ `BaseDetector`, `ABC`, `DriftConfigMixin`

### Constructor

```python
BaseUnivariateDrift(self, x_ref: Union[numpy.ndarray, list], p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, correction: str = 'bonferroni', n_features: Optional[int] = None, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for significance of the statistical test for each feature. If the FDR correction method is used, this corresponds to the acceptable q-value. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. Typically a dimensionality reduction technique. |
| `correction` | `str` | `'bonferroni'` | Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate). |
| `n_features` | `Optional[int]` | `None` | Number of features used in the statistical test. No need to pass it if no preprocessing takes place. In case of a preprocessing step, this can also be inferred automatically but could be more expensive to compute. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. Needs to be provided for text data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `feature_score`

```python
feature_score(x_ref: numpy.ndarray, x: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `numpy.ndarray` |  |  |
| `x` | `numpy.ndarray` |  |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `predict`

```python
predict(x: Union[numpy.ndarray, list], drift_type: str = 'batch', return_p_val: bool = True, return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[numpy.ndarray, int, float]]]
```

Predict whether a batch of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `drift_type` | `str` | `'batch'` | Predict drift at the 'feature' or 'batch' level. For 'batch', the test statistics for each feature are aggregated using the Bonferroni or False Discovery Rate correction (if n_features>1). |
| `return_p_val` | `bool` | `True` | Whether to return feature level p-values. |
| `return_distance` | `bool` | `True` | Whether to return the test statistic between the features of the new batch and reference data. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[numpy.ndarray, int, float]]]`

#### `preprocess`

```python
preprocess(x: Union[numpy.ndarray, list]) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Data preprocessing before computing the drift scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `score`

```python
score(x: Union[numpy.ndarray, list]) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Compute the feature-wise drift score which is the p-value of the

statistical test and the test statistic.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`
