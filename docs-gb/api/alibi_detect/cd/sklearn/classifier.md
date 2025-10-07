# `alibi_detect.cd.sklearn.classifier`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.cd.sklearn.classifier (WARNING)>
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

## `ClassifierDriftSklearn`

_Inherits from:_ `BaseClassifierDrift`, `BaseDetector`, `ABC`

### Constructor

```python
ClassifierDriftSklearn(self, x_ref: numpy.ndarray, model: sklearn.base.ClassifierMixin, p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, preds_type: str = 'probs', binarize_preds: bool = False, train_size: Optional[float] = 0.75, n_folds: Optional[int] = None, retrain_from_scratch: bool = True, seed: int = 0, use_calibration: bool = False, calibration_kwargs: Optional[dict] = None, use_oob: bool = False, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `numpy.ndarray` |  | Data used as reference distribution. |
| `model` | `sklearn.base.ClassifierMixin` |  | Sklearn classification model used for drift detection. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `preds_type` | `str` | `'probs'` | Whether the model outputs 'probs' or 'scores'. |
| `binarize_preds` | `bool` | `False` | Whether to test for discrepancy on soft (e.g. probs/scores) model predictions directly with a K-S test or binarise to 0-1 prediction errors and apply a binomial test. |
| `train_size` | `Optional[float]` | `0.75` | Optional fraction (float between 0 and 1) of the dataset used to train the classifier. The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`. |
| `n_folds` | `Optional[int]` | `None` | Optional number of stratified folds used for training. The model preds are then calculated on all the out-of-fold predictions. This allows to leverage all the reference and test data for drift detection at the expense of longer computation. If both `train_size` and `n_folds` are specified, `n_folds` is prioritized. |
| `retrain_from_scratch` | `bool` | `True` | Whether the classifier should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set. |
| `seed` | `int` | `0` | Optional random seed for fold selection. |
| `use_calibration` | `bool` | `False` | Whether to use calibration. Whether to use calibration. Calibration can be used on top of any model. |
| `calibration_kwargs` | `Optional[dict]` | `None` | Optional additional kwargs for calibration. See https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html for more details. |
| `use_oob` | `bool` | `False` | Whether to use out-of-bag(OOB) predictions. Supported only for `RandomForestClassifier`. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `score`

```python
score(x: Union[numpy.ndarray, list]) -> Tuple[float, float, numpy.ndarray, numpy.ndarray, Union[numpy.ndarray, list], Union[numpy.ndarray, list]]
```

Compute the out-of-fold drift metric such as the accuracy from a classifier

trained to distinguish the reference data from the data to be tested.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[float, float, numpy.ndarray, numpy.ndarray, Union[numpy.ndarray, list], Union[numpy.ndarray, list]]`
