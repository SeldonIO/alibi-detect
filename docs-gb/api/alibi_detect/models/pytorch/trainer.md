# `alibi_detect.models.pytorch.trainer`
## Functions
### `trainer`

```python
trainer(model: Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential], loss_fn: Callable, dataloader: torch.utils.data.dataloader.DataLoader, device: torch.device, optimizer: Callable = <class 'torch.optim.adam.Adam'>, learning_rate: float = 0.001, preprocess_fn: Optional[Callable] = None, epochs: int = 20, reg_loss_fn: Callable = <function <lambda> at 0x292289ca0>, verbose: int = 1) -> None
```

Train PyTorch model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential]` |  | Model to train. |
| `loss_fn` | `Callable` |  | Loss function used for training. |
| `dataloader` | `torch.utils.data.dataloader.DataLoader` |  | PyTorch dataloader. |
| `device` | `torch.device` |  | Device used for training. |
| `optimizer` | `Callable` | `<class 'torch.optim.adam.Adam'>` | Optimizer used for training. |
| `learning_rate` | `float` | `0.001` | Optimizer's learning rate. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Preprocessing function applied to each training batch. |
| `epochs` | `int` | `20` | Number of training epochs. |
| `reg_loss_fn` | `Callable` | `<function <lambda> at 0x292289ca0>` | The regularisation term reg_loss_fn(model) is added to the loss function being optimized. |
| `verbose` | `int` | `1` | Whether to print training progress. |

**Returns**
- Type: `None`
