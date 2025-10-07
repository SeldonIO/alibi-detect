# `alibi_detect.cd.model_uncertainty`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.cd.model_uncertainty (WARNING)>
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

## `ClassifierUncertaintyDrift`

_Inherits from:_ `DriftConfigMixin`

### Constructor

```python
ClassifierUncertaintyDrift(self, x_ref: Union[numpy.ndarray, list], model: Callable, p_val: float = 0.05, x_ref_preprocessed: bool = False, backend: Optional[str] = None, update_x_ref: Optional[Dict[str, int]] = None, preds_type: str = 'probs', uncertainty_type: str = 'entropy', margin_width: float = 0.1, batch_size: int = 32, preprocess_batch_fn: Optional[Callable] = None, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None, tokenizer: Optional[Callable] = None, max_len: Optional[int] = None, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. Should be disjoint from the data the model was trained on for accurate p-values. |
| `model` | `Callable` |  | Classification model outputting class probabilities (or logits) |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `backend` | `Optional[str]` | `None` | Backend to use if model requires batch prediction. Options are 'tensorflow' or 'pytorch'. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preds_type` | `str` | `'probs'` | Type of prediction output by the model. Options are 'probs' (in [0,1]) or 'logits' (in [-inf,inf]). |
| `uncertainty_type` | `str` | `'entropy'` | Method for determining the model's uncertainty for a given instance. Options are 'entropy' or 'margin'. |
| `margin_width` | `float` | `0.1` | Width of the margin if uncertainty_type = 'margin'. The model is considered uncertain on an instance if the highest two class probabilities it assigns to the instance differ by less than margin_width. |
| `batch_size` | `int` | `32` | Batch size used to evaluate model. Only relevant when backend has been specified for batch prediction. |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` | Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed by the model. |
| `device` | `Union[Literal[cuda, gpu, cpu], ForwardRef('torch.device'), None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. Only relevant for 'pytorch' backend. |
| `tokenizer` | `Optional[Callable]` | `None` | Optional tokenizer for NLP models. |
| `max_len` | `Optional[int]` | `None` | Optional max token length for NLP models. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `predict`

```python
predict(x: Union[numpy.ndarray, list], return_p_val: bool = True, return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[numpy.ndarray, int, float]]]
```

Predict whether a batch of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `return_p_val` | `bool` | `True` | Whether to return the p-value of the test. |
| `return_distance` | `bool` | `True` | Whether to return the corresponding test statistic (K-S for 'entropy', Chi2 for 'margin'). |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[numpy.ndarray, int, float]]]`

## `RegressorUncertaintyDrift`

_Inherits from:_ `DriftConfigMixin`

### Constructor

```python
RegressorUncertaintyDrift(self, x_ref: Union[numpy.ndarray, list], model: Callable, p_val: float = 0.05, x_ref_preprocessed: bool = False, backend: Optional[str] = None, update_x_ref: Optional[Dict[str, int]] = None, uncertainty_type: str = 'mc_dropout', n_evals: int = 25, batch_size: int = 32, preprocess_batch_fn: Optional[Callable] = None, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None, tokenizer: Optional[Callable] = None, max_len: Optional[int] = None, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. Should be disjoint from the data the model was trained on for accurate p-values. |
| `model` | `Callable` |  | Regression model outputting class probabilities (or logits) |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `backend` | `Optional[str]` | `None` | Backend to use if model requires batch prediction. Options are 'tensorflow' or 'pytorch'. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `uncertainty_type` | `str` | `'mc_dropout'` | Method for determining the model's uncertainty for a given instance. Options are 'mc_dropout' or 'ensemble'. The former should output a scalar per instance. The latter should output a vector of predictions per instance. |
| `n_evals` | `int` | `25` | The number of times to evaluate the model under different dropout configurations. Only relevant when using the 'mc_dropout' uncertainty type. |
| `batch_size` | `int` | `32` | Batch size used to evaluate model. Only relevant when backend has been specified for batch prediction. |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` | Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed by the model. |
| `device` | `Union[Literal[cuda, gpu, cpu], ForwardRef('torch.device'), None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. Only relevant for 'pytorch' backend. |
| `tokenizer` | `Optional[Callable]` | `None` | Optional tokenizer for NLP models. |
| `max_len` | `Optional[int]` | `None` | Optional max token length for NLP models. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `predict`

```python
predict(x: Union[numpy.ndarray, list], return_p_val: bool = True, return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[numpy.ndarray, int, float]]]
```

Predict whether a batch of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `return_p_val` | `bool` | `True` | Whether to return the p-value of the test. |
| `return_distance` | `bool` | `True` | Whether to return the K-S test statistic |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[numpy.ndarray, int, float]]]`
