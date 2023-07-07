import torch


def hinge_loss(preds: torch.Tensor) -> torch.Tensor:
    "L(pred) = max(0, 1-pred) averaged over multiple preds"
    linear_inds = preds < 1
    return (((1 - preds)*linear_inds).sum(0))/len(preds)
