# `alibi_detect.utils.pytorch.losses`
## Functions
### `hinge_loss`

```python
hinge_loss(preds: torch.Tensor) -> torch.Tensor
```

L(pred) = max(0, 1-pred) averaged over multiple preds

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `preds` | `torch.Tensor` |  |  |

**Returns**
- Type: `torch.Tensor`
