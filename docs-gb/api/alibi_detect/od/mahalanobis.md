# `alibi_detect.od.mahalanobis`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.od.mahalanobis (WARNING)>
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

### `EPSILON`
```python
EPSILON: float = 1e-08
```
Convert a string or number to a floating point number, if possible.

## `Mahalanobis`

_Inherits from:_ `BaseDetector`, `FitMixin`, `ThresholdMixin`, `ABC`

### Constructor

```python
Mahalanobis(self, threshold: float = None, n_components: int = 3, std_clip: int = 3, start_clip: int = 100, max_n: int = None, cat_vars: dict = None, ohe: bool = False, data_type: str = 'tabular') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold` | `Optional[float]` | `None` | Mahalanobis distance threshold used to classify outliers. |
| `n_components` | `int` | `3` | Number of principal components used. |
| `std_clip` | `int` | `3` | Feature-wise stdev used to clip the observations before updating the mean and cov. |
| `start_clip` | `int` | `100` | Number of observations before clipping is applied. |
| `max_n` | `Optional[int]` | `None` | Algorithm behaves as if it has seen at most max_n points. |
| `cat_vars` | `Optional[dict]` | `None` | Dict with as keys the categorical columns and as values the number of categories per categorical variable. |
| `ohe` | `bool` | `False` | Whether the categorical variables are one-hot encoded (OHE) or not. If not OHE, they are assumed to have ordinal encodings. |
| `data_type` | `str` | `'tabular'` | Optionally specifiy the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `cat2num`

```python
cat2num(X: numpy.ndarray) -> numpy.ndarray
```

Convert categorical variables to numerical values.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances to analyze. |

**Returns**
- Type: `numpy.ndarray`

#### `fit`

```python
fit(X: numpy.ndarray, y: Optional[numpy.ndarray] = None, d_type: str = 'abdm', w: Optional[float] = None, disc_perc: list = [25, 50, 75], standardize_cat_vars: bool = True, feature_range: tuple = (-10000000000.0, 10000000000.0), smooth: float = 1.0, center: bool = True) -> None
```

If categorical variables are present, then transform those to numerical values.

This step is not necessary in the absence of categorical variables.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances used to infer distances between categories from. |
| `y` | `Optional[numpy.ndarray]` | `None` | Model class predictions or ground truth labels for X. Used for 'mvdm' and 'abdm-mvdm' pairwise distance metrics. Note that this is only compatible with classification problems. For regression problems, use the 'abdm' distance metric. |
| `d_type` | `str` | `'abdm'` | Pairwise distance metric used for categorical variables. Currently, 'abdm', 'mvdm' and 'abdm-mvdm' are supported. 'abdm' infers context from the other variables while 'mvdm' uses the model predictions. 'abdm-mvdm' is a weighted combination of the two metrics. |
| `w` | `Optional[float]` | `None` | Weight on 'abdm' (between 0. and 1.) distance if d_type equals 'abdm-mvdm'. |
| `disc_perc` | `list` | `[25, 50, 75]` | List with percentiles used in binning of numerical features used for the 'abdm' and 'abdm-mvdm' pairwise distance measures. |
| `standardize_cat_vars` | `bool` | `True` | Standardize numerical values of categorical variables if True. |
| `feature_range` | `tuple` | `(-10000000000.0, 10000000000.0)` | Tuple with min and max ranges to allow for perturbed instances. Min and max ranges can be floats or numpy arrays with dimension (1x nb of features) for feature-wise ranges. |
| `smooth` | `float` | `1.0` | Smoothing exponent between 0 and 1 for the distances. Lower values of l will smooth the difference in distance metric between different features. |
| `center` | `bool` | `True` | Whether to center the scaled distance measures. If False, the min distance for each feature except for the feature with the highest raw max distance will be the lower bound of the feature range, but the upper bound will be below the max feature range. |

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
