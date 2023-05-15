from torch import nn
import torch


class GMMModel(nn.Module):
    def __init__(self, n_components: int, dim: int) -> None:
        """Gaussian Mixture Model (GMM).

        Parameters
        ----------
        n_components
            The number of mixture components.
        dim
            The dimensionality of the data.
        """
        super().__init__()
        self.weight_logits = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, dim))
        self.inv_cov_factor = nn.Parameter(torch.randn(n_components, dim, dim)/10)

    @property
    def _inv_cov(self) -> torch.Tensor:
        return torch.bmm(self.inv_cov_factor, self.inv_cov_factor.transpose(1, 2))

    @property
    def _weights(self) -> torch.Tensor:
        return nn.functional.softmax(self.weight_logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log-likelihood of the data.

        Parameters
        ----------
        x
            Data to score.
        """
        det = torch.linalg.det(self._inv_cov)  # Note det(A^-1)=1/det(A)
        to_means = x[:, None, :] - self.means[None, :, :]
        likelihood = ((-0.5 * (
            torch.einsum('bke,bke->bk', (torch.einsum('bkd,kde->bke', to_means, self._inv_cov), to_means))
        )).exp()*det[None, :]*self._weights[None, :]).sum(-1)
        return -likelihood.log()
