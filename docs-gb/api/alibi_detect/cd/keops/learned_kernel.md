# `alibi_detect.cd.keops.learned_kernel`
## `LearnedKernelDriftKeops`

_Inherits from:_ `BaseLearnedKernelDrift`, `BaseDetector`, `ABC`

### Constructor

```python
LearnedKernelDriftKeops(self, x_ref: Union[numpy.ndarray, list], kernel: Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential], p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, n_permutations: int = 100, batch_size_permutations: int = 1000000, var_reg: float = 1e-05, reg_loss_fn: Callable = <function LearnedKernelDriftKeops.<lambda> at 0x2922f7550>, train_size: Optional[float] = 0.75, retrain_from_scratch: bool = True, optimizer: torch.optim.optimizer.Optimizer = <class 'torch.optim.adam.Adam'>, learning_rate: float = 0.001, batch_size: int = 32, batch_size_predict: int = 1000000, preprocess_batch_fn: Optional[Callable] = None, epochs: int = 3, num_workers: int = 0, verbose: int = 0, train_kwargs: Optional[dict] = None, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None, dataset: Callable = <class 'alibi_detect.utils.pytorch.data.TorchDataset'>, dataloader: Callable = <class 'torch.utils.data.dataloader.DataLoader'>, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `kernel` | `Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential]` |  | Trainable PyTorch module that returns a similarity between two instances. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before applying the kernel. |
| `n_permutations` | `int` | `100` | The number of permutations to use in the permutation test once the MMD has been computed. |
| `batch_size_permutations` | `int` | `1000000` | KeOps computes the n_permutations of the MMD^2 statistics in chunks of batch_size_permutations. |
| `var_reg` | `float` | `1e-05` | Constant added to the estimated variance of the MMD for stability. |
| `reg_loss_fn` | `Callable` | `<function LearnedKernelDriftKeops.<lambda> at 0x2922f7550>` | The regularisation term reg_loss_fn(kernel) is added to the loss function being optimized. |
| `train_size` | `Optional[float]` | `0.75` | Optional fraction (float between 0 and 1) of the dataset used to train the kernel. The drift is detected on `1 - train_size`. |
| `retrain_from_scratch` | `bool` | `True` | Whether the kernel should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set. |
| `optimizer` | `torch.optim.optimizer.Optimizer` | `<class 'torch.optim.adam.Adam'>` | Optimizer used during training of the kernel. |
| `learning_rate` | `float` | `0.001` | Learning rate used by optimizer. |
| `batch_size` | `int` | `32` | Batch size used during training of the kernel. |
| `batch_size_predict` | `int` | `1000000` | Batch size used for the trained drift detector predictions. |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` | Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed by the kernel. |
| `epochs` | `int` | `3` | Number of training epochs for the kernel. Corresponds to the smaller of the reference and test sets. |
| `num_workers` | `int` | `0` | Number of workers for the dataloader. The default (`num_workers=0`) means multi-process data loading is disabled. Setting `num_workers>0` may be unreliable on Windows. |
| `verbose` | `int` | `0` | Verbosity level during the training of the kernel. 0 is silent, 1 a progress bar. |
| `train_kwargs` | `Optional[dict]` | `None` | Optional additional kwargs when training the kernel. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. Relevant for 'pytorch' and 'keops' backends. |
| `dataset` | `Callable` | `<class 'alibi_detect.utils.pytorch.data.TorchDataset'>` | Dataset object used during training. |
| `dataloader` | `Callable` | `<class 'torch.utils.data.dataloader.DataLoader'>` | Dataloader object used during training. Only relevant for 'pytorch' backend. |
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
trainer(j_hat: alibi_detect.cd.keops.learned_kernel.LearnedKernelDriftKeops.JHat, dataloaders: Tuple[torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader], device: torch.device, optimizer: Callable = <class 'torch.optim.adam.Adam'>, learning_rate: float = 0.001, preprocess_fn: Optional[Callable] = None, epochs: int = 20, reg_loss_fn: Callable = <function LearnedKernelDriftKeops.<lambda> at 0x2922f78b0>, verbose: int = 1) -> None
```

Train the kernel to maximise an estimate of test power using minibatch gradient descent.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `j_hat` | `alibi_detect.cd.keops.learned_kernel.LearnedKernelDriftKeops.JHat` |  |  |
| `dataloaders` | `Tuple[torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader]` |  |  |
| `device` | `torch.device` |  |  |
| `optimizer` | `Callable` | `<class 'torch.optim.adam.Adam'>` |  |
| `learning_rate` | `float` | `0.001` |  |
| `preprocess_fn` | `Optional[Callable]` | `None` |  |
| `epochs` | `int` | `20` |  |
| `reg_loss_fn` | `Callable` | `<function LearnedKernelDriftKeops.<lambda> at 0x2922f78b0>` |  |
| `verbose` | `int` | `1` |  |

**Returns**
- Type: `None`
