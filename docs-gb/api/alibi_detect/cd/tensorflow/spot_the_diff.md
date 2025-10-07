# `alibi_detect.cd.tensorflow.spot_the_diff`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.cd.tensorflow.spot_the_diff (WARNING)>
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

## `SpotTheDiffDriftTF`

### Constructor

```python
SpotTheDiffDriftTF(self, x_ref: numpy.ndarray, p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_fn: Optional[Callable] = None, kernel: Optional[keras.src.models.model.Model] = None, n_diffs: int = 1, initial_diffs: Optional[numpy.ndarray] = None, l1_reg: float = 0.01, binarize_preds: bool = False, train_size: Optional[float] = 0.75, n_folds: Optional[int] = None, retrain_from_scratch: bool = True, seed: int = 0, optimizer: <module 'tensorflow.keras.optimizers' from '/Users/paul.bridi/Projects/alibi/venv/lib/python3.9/site-packages/keras/_tf_keras/keras/optimizers/__init__.py'> = <class 'keras.src.optimizers.adam.Adam'>, learning_rate: float = 0.001, batch_size: int = 32, preprocess_batch_fn: Optional[Callable] = None, epochs: int = 3, verbose: int = 0, train_kwargs: Optional[dict] = None, dataset: Callable = <class 'alibi_detect.utils.tensorflow.data.TFDataset'>, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `numpy.ndarray` |  | Data used as reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `kernel` | `Optional[keras.src.models.model.Model]` | `None` | Differentiable TensorFlow model used to define similarity between instances, defaults to Gaussian RBF. |
| `n_diffs` | `int` | `1` | The number of test locations to use, each corresponding to an interpretable difference. |
| `initial_diffs` | `Optional[numpy.ndarray]` | `None` | Array used to initialise the diffs that will be learned. Defaults to Gaussian for each feature with equal variance to that of reference data. |
| `l1_reg` | `float` | `0.01` | Strength of l1 regularisation to apply to the differences. |
| `binarize_preds` | `bool` | `False` | Whether to test for discrepency on soft (e.g. probs/logits) model predictions directly with a K-S test or binarise to 0-1 prediction errors and apply a binomial test. |
| `train_size` | `Optional[float]` | `0.75` | Optional fraction (float between 0 and 1) of the dataset used to train the classifier. The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`. |
| `n_folds` | `Optional[int]` | `None` | Optional number of stratified folds used for training. The model preds are then calculated on all the out-of-fold instances. This allows to leverage all the reference and test data for drift detection at the expense of longer computation. If both `train_size` and `n_folds` are specified, `n_folds` is prioritized. |
| `retrain_from_scratch` | `bool` | `True` | Whether the classifier should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set. |
| `seed` | `int` | `0` | Optional random seed for fold selection. |
| `optimizer` | `.tensorflow.keras.optimizers` | `<class 'keras.src.optimizers.adam.Adam'>` | Optimizer used during training of the classifier. |
| `learning_rate` | `float` | `0.001` | Learning rate used by optimizer. |
| `batch_size` | `int` | `32` | Batch size used during training of the classifier. |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` | Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed by the model. |
| `epochs` | `int` | `3` | Number of training epochs for the classifier for each (optional) fold. |
| `verbose` | `int` | `0` | Verbosity level during the training of the classifier. 0 is silent, 1 a progress bar. |
| `train_kwargs` | `Optional[dict]` | `None` | Optional additional kwargs when fitting the classifier. |
| `dataset` | `Callable` | `<class 'alibi_detect.utils.tensorflow.data.TFDataset'>` | Dataset object used during training. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `predict`

```python
predict(x: numpy.ndarray, return_p_val: bool = True, return_distance: bool = True, return_probs: bool = True, return_model: bool = False) -> Dict[str, Dict[str, Union[str, int, float, Callable]]]
```

Predict whether a batch of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Batch of instances. |
| `return_p_val` | `bool` | `True` | Whether to return the p-value of the test. |
| `return_distance` | `bool` | `True` | Whether to return a notion of strength of the drift. K-S test stat if binarize_preds=False, otherwise relative error reduction. |
| `return_probs` | `bool` | `True` | Whether to return the instance level classifier probabilities for the reference and test data (0=reference data, 1=test data). |
| `return_model` | `bool` | `False` | Whether to return the updated model trained to discriminate reference and test instances. |

**Returns**
- Type: `Dict[str, Dict[str, Union[str, int, float, Callable]]]`
