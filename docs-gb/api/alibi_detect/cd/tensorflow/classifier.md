# `alibi_detect.cd.tensorflow.classifier`
## `ClassifierDriftTF`

_Inherits from:_ `BaseClassifierDrift`, `BaseDetector`, `ABC`

### Constructor

```python
ClassifierDriftTF(self, x_ref: numpy.ndarray, model: keras.src.models.model.Model, p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, preds_type: str = 'probs', binarize_preds: bool = False, reg_loss_fn: Callable = <function ClassifierDriftTF.<lambda> at 0x28fde7430>, train_size: Optional[float] = 0.75, n_folds: Optional[int] = None, retrain_from_scratch: bool = True, seed: int = 0, optimizer: Union[ForwardRef('tf.keras.optimizers.Optimizer'), ForwardRef('tf.keras.optimizers.legacy.Optimizer'), Type[ForwardRef('tf.keras.optimizers.Optimizer')], Type[ForwardRef('tf.keras.optimizers.legacy.Optimizer')]] = <class 'keras.src.optimizers.adam.Adam'>, learning_rate: float = 0.001, batch_size: int = 32, preprocess_batch_fn: Optional[Callable] = None, epochs: int = 3, verbose: int = 0, train_kwargs: Optional[dict] = None, dataset: Callable = <class 'alibi_detect.utils.tensorflow.data.TFDataset'>, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `numpy.ndarray` |  | Data used as reference distribution. |
| `model` | `keras.src.models.model.Model` |  | TensorFlow classification model used for drift detection. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `preds_type` | `str` | `'probs'` | Whether the model outputs 'probs' or 'logits'. |
| `binarize_preds` | `bool` | `False` | Whether to test for discrepency on soft (e.g. prob/log-prob) model predictions directly with a K-S test or binarise to 0-1 prediction errors and apply a binomial test. |
| `reg_loss_fn` | `Callable` | `<function ClassifierDriftTF.<lambda> at 0x28fde7430>` | The regularisation term reg_loss_fn(model) is added to the loss function being optimized. |
| `train_size` | `Optional[float]` | `0.75` | Optional fraction (float between 0 and 1) of the dataset used to train the classifier. The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`. |
| `n_folds` | `Optional[int]` | `None` | Optional number of stratified folds used for training. The model preds are then calculated on all the out-of-fold predictions. This allows to leverage all the reference and test data for drift detection at the expense of longer computation. If both `train_size` and `n_folds` are specified, `n_folds` is prioritized. |
| `retrain_from_scratch` | `bool` | `True` | Whether the classifier should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set. |
| `seed` | `int` | `0` | Optional random seed for fold selection. |
| `optimizer` | `Union[keras.src.optimizers.optimizer.Optimizer, keras.src.optimizers.LegacyOptimizerWarning, type[keras.src.optimizers.optimizer.Optimizer], type[keras.src.optimizers.LegacyOptimizerWarning]]` | `<class 'keras.src.optimizers.adam.Adam'>` | Optimizer used during training of the classifier. |
| `learning_rate` | `float` | `0.001` | Learning rate used by optimizer. |
| `batch_size` | `int` | `32` | Batch size used during training of the classifier. |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` | Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed by the model. |
| `epochs` | `int` | `3` | Number of training epochs for the classifier for each (optional) fold. |
| `verbose` | `int` | `0` | Verbosity level during the training of the classifier. 0 is silent, 1 a progress bar and 2 prints the statistics after each epoch. |
| `train_kwargs` | `Optional[dict]` | `None` | Optional additional kwargs when fitting the classifier. |
| `dataset` | `Callable` | `<class 'alibi_detect.utils.tensorflow.data.TFDataset'>` | Dataset object used during training. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `score`

```python
score(x: numpy.ndarray) -> Tuple[float, float, numpy.ndarray, numpy.ndarray, Union[numpy.ndarray, list], Union[numpy.ndarray, list]]
```

Compute the out-of-fold drift metric such as the accuracy from a classifier

trained to distinguish the reference data from the data to be tested.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Batch of instances. |

**Returns**
- Type: `Tuple[float, float, numpy.ndarray, numpy.ndarray, Union[numpy.ndarray, list], Union[numpy.ndarray, list]]`
