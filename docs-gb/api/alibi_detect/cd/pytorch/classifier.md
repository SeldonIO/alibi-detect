# `alibi_detect.cd.pytorch.classifier`
## `ClassifierDriftTorch`

_Inherits from:_ `BaseClassifierDrift`, `BaseDetector`, `ABC`

### Constructor

```python
ClassifierDriftTorch(self, x_ref: Union[numpy.ndarray, list], model: Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential], p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, preds_type: str = 'probs', binarize_preds: bool = False, reg_loss_fn: Callable = <function ClassifierDriftTorch.<lambda> at 0x28fe6ed30>, train_size: Optional[float] = 0.75, n_folds: Optional[int] = None, retrain_from_scratch: bool = True, seed: int = 0, optimizer: Callable = <class 'torch.optim.adam.Adam'>, learning_rate: float = 0.001, batch_size: int = 32, preprocess_batch_fn: Optional[Callable] = None, epochs: int = 3, verbose: int = 0, train_kwargs: Optional[dict] = None, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None, dataset: Callable = <class 'alibi_detect.utils.pytorch.data.TorchDataset'>, dataloader: Callable = <class 'torch.utils.data.dataloader.DataLoader'>, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `model` | `Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential]` |  | PyTorch classification model used for drift detection. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `preds_type` | `str` | `'probs'` | Whether the model outputs 'probs' or 'logits' |
| `binarize_preds` | `bool` | `False` | Whether to test for discrepency on soft (e.g. probs/logits) model predictions directly with a K-S test or binarise to 0-1 prediction errors and apply a binomial test. |
| `reg_loss_fn` | `Callable` | `<function ClassifierDriftTorch.<lambda> at 0x28fe6ed30>` | The regularisation term reg_loss_fn(model) is added to the loss function being optimized. |
| `train_size` | `Optional[float]` | `0.75` | Optional fraction (float between 0 and 1) of the dataset used to train the classifier. The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`. |
| `n_folds` | `Optional[int]` | `None` | Optional number of stratified folds used for training. The model preds are then calculated on all the out-of-fold predictions. This allows to leverage all the reference and test data for drift detection at the expense of longer computation. If both `train_size` and `n_folds` are specified, `n_folds` is prioritized. |
| `retrain_from_scratch` | `bool` | `True` | Whether the classifier should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set. |
| `seed` | `int` | `0` | Optional random seed for fold selection. |
| `optimizer` | `Callable` | `<class 'torch.optim.adam.Adam'>` | Optimizer used during training of the classifier. |
| `learning_rate` | `float` | `0.001` | Learning rate used by optimizer. |
| `batch_size` | `int` | `32` | Batch size used during training of the classifier. |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` | Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed by the model. |
| `epochs` | `int` | `3` | Number of training epochs for the classifier for each (optional) fold. |
| `verbose` | `int` | `0` | Verbosity level during the training of the classifier. 0 is silent, 1 a progress bar. |
| `train_kwargs` | `Optional[dict]` | `None` | Optional additional kwargs when fitting the classifier. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |
| `dataset` | `Callable` | `<class 'alibi_detect.utils.pytorch.data.TorchDataset'>` | Dataset object used during training. |
| `dataloader` | `Callable` | `<class 'torch.utils.data.dataloader.DataLoader'>` | Dataloader object used during training. |
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
