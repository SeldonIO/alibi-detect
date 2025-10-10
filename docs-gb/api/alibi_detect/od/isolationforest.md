# `alibi_detect.od.isolationforest`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.od.isolationforest (WARNING)>
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

## `IForest`

_Inherits from:_ `BaseDetector`, `FitMixin`, `ThresholdMixin`, `ABC`

### Constructor

```python
IForest(self, threshold: float = None, n_estimators: int = 100, max_samples: Union[str, int, float] = 'auto', max_features: Union[int, float] = 1.0, bootstrap: bool = False, n_jobs: int = 1, data_type: str = 'tabular') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold` | `Optional[float]` | `None` | Threshold used for outlier score to determine outliers. |
| `n_estimators` | `int` | `100` | Number of base estimators in the ensemble. |
| `max_samples` | `Union[str, int, float]` | `'auto'` | Number of samples to draw from the training data to train each base estimator. If int, draw 'max_samples' samples. If float, draw 'max_samples * number of features' samples. If 'auto', max_samples = min(256, number of samples) |
| `max_features` | `Union[int, float]` | `1.0` | Number of features to draw from the training data to train each base estimator. If int, draw 'max_features' features. If float, draw 'max_features * number of features' features. |
| `bootstrap` | `bool` | `False` | Whether to fit individual trees on random subsets of the training data, sampled with replacement. |
| `n_jobs` | `int` | `1` | Number of jobs to run in parallel for 'fit' and 'predict'. |
| `data_type` | `str` | `'tabular'` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `fit`

```python
fit(X: numpy.ndarray, sample_weight: Optional[numpy.ndarray] = None) -> None
```

Fit isolation forest.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Training batch. |
| `sample_weight` | `Optional[numpy.ndarray]` | `None` | Sample weights. |

**Returns**
- Type: `None`

#### `infer_threshold`

```python
infer_threshold(X: numpy.ndarray, threshold_perc: float = 95.0) -> None
```

Update threshold by a value inferred from the percentage of instances considered to be

outliers in a sample of the dataset.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `threshold_perc` | `float` | `95.0` | Percentage of X considered to be normal based on the outlier score. |

**Returns**
- Type: `None`

#### `predict`

```python
predict(X: numpy.ndarray, return_instance_score: bool = True) -> Dict[Dict[str, str], Dict[numpy.ndarray, numpy.ndarray]]
```

Compute outlier scores and transform into outlier predictions.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `return_instance_score` | `bool` | `True` | Whether to return instance level outlier scores. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[numpy.ndarray, numpy.ndarray]]`

#### `score`

```python
score(X: numpy.ndarray) -> numpy.ndarray
```

Compute outlier scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances to analyze. |

**Returns**
- Type: `numpy.ndarray`
