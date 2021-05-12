import torch

ten0 = torch.tensor([1,2,3])
ten1 = torch.cat([ten0, ten0], 0)