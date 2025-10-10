# `alibi_detect.cd.classifier`
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

## `ClassifierDrift`

_Inherits from:_ `DriftConfigMixin`

### Constructor

```python
ClassifierDrift(self, x_ref: Union[numpy.ndarray, list], model: Union[sklearn.base.ClassifierMixin, Callable], backend: str = 'tensorflow', p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, preds_type: str = 'probs', binarize_preds: bool = False, reg_loss_fn: Callable = <function ClassifierDrift.<lambda> at 0x28fe6e9d0>, train_size: Optional[float] = 0.75, n_folds: Optional[int] = None, retrain_from_scratch: bool = True, seed: int = 0, optimizer: Optional[Callable] = None, learning_rate: float = 0.001, batch_size: int = 32, preprocess_batch_fn: Optional[Callable] = None, epochs: int = 3, verbose: int = 0, train_kwargs: Optional[dict] = None, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None, dataset: Optional[Callable] = None, dataloader: Optional[Callable] = None, input_shape: Optional[tuple] = None, use_calibration: bool = False, calibration_kwargs: Optional[dict] = None, use_oob: bool = False, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `model` | `Union[sklearn.base.ClassifierMixin, Callable]` |  | PyTorch, TensorFlow or Sklearn classification model used for drift detection. |
| `backend` | `str` | `'tensorflow'` | Backend used for the training loop implementation. Supported: 'tensorflow' | 'pytorch' | 'sklearn'. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last `n` instances seen by the detector or via reservoir sampling with size `n`. For the former, the parameter equals `{'last': n}` while for reservoir sampling `{'reservoir_sampling': n}` is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `preds_type` | `str` | `'probs'` | Whether the model outputs 'probs' (probabilities - for 'tensorflow', 'pytorch', 'sklearn' models), 'logits' (for 'pytorch', 'tensorflow' models), 'scores' (for 'sklearn' models if `decision_function` is supported). |
| `binarize_preds` | `bool` | `False` | Whether to test for discrepancy on soft  (e.g. probs/logits/scores) model predictions directly with a K-S test or binarise to 0-1 prediction errors and apply a binomial test. |
| `reg_loss_fn` | `Callable` | `<function ClassifierDrift.<lambda> at 0x28fe6e9d0>` | The regularisation term `reg_loss_fn(model)` is added to the loss function being optimized. Only relevant for 'tensorflow` and 'pytorch' backends. |
| `train_size` | `Optional[float]` | `0.75` | Optional fraction (float between 0 and 1) of the dataset used to train the classifier. The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`. |
| `n_folds` | `Optional[int]` | `None` | Optional number of stratified folds used for training. The model preds are then calculated on all the out-of-fold instances. This allows to leverage all the reference and test data for drift detection at the expense of longer computation. If both `train_size` and `n_folds` are specified, `n_folds` is prioritized. |
| `retrain_from_scratch` | `bool` | `True` | Whether the classifier should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set. |
| `seed` | `int` | `0` | Optional random seed for fold selection. |
| `optimizer` | `Optional[Callable]` | `None` | Optimizer used during training of the classifier. Only relevant for 'tensorflow' and 'pytorch' backends. |
| `learning_rate` | `float` | `0.001` | Learning rate used by optimizer. Only relevant for 'tensorflow' and 'pytorch' backends. |
| `batch_size` | `int` | `32` | Batch size used during training of the classifier. Only relevant for 'tensorflow' and 'pytorch' backends. |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` | Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed by the model. Only relevant for 'tensorflow' and 'pytorch' backends. |
| `epochs` | `int` | `3` | Number of training epochs for the classifier for each (optional) fold. Only relevant for 'tensorflow' and 'pytorch' backends. |
| `verbose` | `int` | `0` | Verbosity level during the training of the classifier. 0 is silent, 1 a progress bar. Only relevant for 'tensorflow' and 'pytorch' backends. |
| `train_kwargs` | `Optional[dict]` | `None` | Optional additional kwargs when fitting the classifier. Only relevant for 'tensorflow' and 'pytorch' backends. |
| `device` | `Union[Literal[cuda, gpu, cpu], ForwardRef('torch.device'), None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. Only relevant for 'pytorch' backend. |
| `dataset` | `Optional[Callable]` | `None` | Dataset object used during training. Only relevant for 'tensorflow' and 'pytorch' backends. |
| `dataloader` | `Optional[Callable]` | `None` | Dataloader object used during training. Only relevant for 'pytorch' backend. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `use_calibration` | `bool` | `False` | Whether to use calibration. Calibration can be used on top of any model. Only relevant for 'sklearn' backend. |
| `calibration_kwargs` | `Optional[dict]` | `None` | Optional additional kwargs for calibration. Only relevant for 'sklearn' backend. See https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html for more details. |
| `use_oob` | `bool` | `False` | Whether to use out-of-bag(OOB) predictions. Supported only for `RandomForestClassifier`. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

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
| `return_probs` | `bool` | `True` | Whether to return the instance level classifier probabilities for the reference and test data (0=reference data, 1=test data). |
| `return_model` | `bool` | `True` | Whether to return the updated model trained to discriminate reference and test instances. |

**Returns**
- Type: `Dict[str, Dict[str, Union[str, int, float, Callable]]]`
