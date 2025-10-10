# `alibi_detect.cd.tensorflow.learned_kernel`
## `LearnedKernelDriftTF`

_Inherits from:_ `BaseLearnedKernelDrift`, `BaseDetector`, `ABC`

### Constructor

```python
LearnedKernelDriftTF(self, x_ref: Union[numpy.ndarray, list], kernel: keras.src.models.model.Model, p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, n_permutations: int = 100, var_reg: float = 1e-05, reg_loss_fn: Callable = <function LearnedKernelDriftTF.<lambda> at 0x28fe7e0d0>, train_size: Optional[float] = 0.75, retrain_from_scratch: bool = True, optimizer: Union[ForwardRef('tf.keras.optimizers.Optimizer'), ForwardRef('tf.keras.optimizers.legacy.Optimizer'), Type[ForwardRef('tf.keras.optimizers.Optimizer')], Type[ForwardRef('tf.keras.optimizers.legacy.Optimizer')]] = <class 'keras.src.optimizers.adam.Adam'>, learning_rate: float = 0.001, batch_size: int = 32, batch_size_predict: int = 32, preprocess_batch_fn: Optional[Callable] = None, epochs: int = 3, verbose: int = 0, train_kwargs: Optional[dict] = None, dataset: Callable = <class 'alibi_detect.utils.tensorflow.data.TFDataset'>, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `kernel` | `keras.src.models.model.Model` |  | Trainable TensorFlow model that returns a similarity between two instances. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before applying the kernel. |
| `n_permutations` | `int` | `100` | The number of permutations to use in the permutation test once the MMD has been computed. |
| `var_reg` | `float` | `1e-05` | Constant added to the estimated variance of the MMD for stability. |
| `reg_loss_fn` | `Callable` | `<function LearnedKernelDriftTF.<lambda> at 0x28fe7e0d0>` | The regularisation term reg_loss_fn(kernel) is added to the loss function being optimized. |
| `train_size` | `Optional[float]` | `0.75` | Optional fraction (float between 0 and 1) of the dataset used to train the kernel. The drift is detected on `1 - train_size`. |
| `retrain_from_scratch` | `bool` | `True` | Whether the kernel should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set. |
| `optimizer` | `Union[keras.src.optimizers.optimizer.Optimizer, keras.src.optimizers.LegacyOptimizerWarning, type[keras.src.optimizers.optimizer.Optimizer], type[keras.src.optimizers.LegacyOptimizerWarning]]` | `<class 'keras.src.optimizers.adam.Adam'>` | Optimizer used during training of the kernel. |
| `learning_rate` | `float` | `0.001` | Learning rate used by optimizer. |
| `batch_size` | `int` | `32` | Batch size used during training of the kernel. |
| `batch_size_predict` | `int` | `32` | Batch size used for the trained drift detector predictions. |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` | Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed by the kernel. |
| `epochs` | `int` | `3` | Number of training epochs for the kernel. Corresponds to the smaller of the reference and test sets. |
| `verbose` | `int` | `0` | Verbosity level during the training of the kernel. 0 is silent, 1 a progress bar. |
| `train_kwargs` | `Optional[dict]` | `None` | Optional additional kwargs when training the kernel. |
| `dataset` | `Callable` | `<class 'alibi_detect.utils.tensorflow.data.TFDataset'>` | Dataset object used during training. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `score`

```python
score(x: Union[numpy.ndarray, list]) -> Tuple[float, float, float]
```

Compute the p-value resulting from a permutation test using the maximum mean discrepancy

as a distance measure between the reference data and the data to be tested. The kernel
used within the MMD is first trained to maximise an estimate of the resulting test power.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[float, float, float]`

#### `trainer`

```python
trainer(j_hat: alibi_detect.cd.tensorflow.learned_kernel.LearnedKernelDriftTF.JHat, datasets: Tuple[keras.src.trainers.data_adapters.py_dataset_adapter.PyDataset, keras.src.trainers.data_adapters.py_dataset_adapter.PyDataset], optimizer: Union[keras.src.optimizers.optimizer.Optimizer, keras.src.optimizers.LegacyOptimizerWarning, type[keras.src.optimizers.optimizer.Optimizer], type[keras.src.optimizers.LegacyOptimizerWarning]] = <class 'keras.src.optimizers.adam.Adam'>, learning_rate: float = 0.001, preprocess_fn: Optional[Callable] = None, epochs: int = 20, reg_loss_fn: Callable = <function LearnedKernelDriftTF.<lambda> at 0x28fe7e430>, verbose: int = 1) -> None
```

Train the kernel to maximise an estimate of test power using minibatch gradient descent.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `j_hat` | `alibi_detect.cd.tensorflow.learned_kernel.LearnedKernelDriftTF.JHat` |  |  |
| `datasets` | `Tuple[keras.src.trainers.data_adapters.py_dataset_adapter.PyDataset, keras.src.trainers.data_adapters.py_dataset_adapter.PyDataset]` |  |  |
| `optimizer` | `Union[keras.src.optimizers.optimizer.Optimizer, keras.src.optimizers.LegacyOptimizerWarning, type[keras.src.optimizers.optimizer.Optimizer], type[keras.src.optimizers.LegacyOptimizerWarning]]` | `<class 'keras.src.optimizers.adam.Adam'>` |  |
| `learning_rate` | `float` | `0.001` |  |
| `preprocess_fn` | `Optional[Callable]` | `None` |  |
| `epochs` | `int` | `20` |  |
| `reg_loss_fn` | `Callable` | `<function LearnedKernelDriftTF.<lambda> at 0x28fe7e430>` |  |
| `verbose` | `int` | `1` |  |

**Returns**
- Type: `None`
