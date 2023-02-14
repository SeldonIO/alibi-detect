from typing import Callable, Optional, Union, Dict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from alibi_detect.utils.pytorch.data import TorchDataset
from alibi_detect.od.pytorch.base import TorchOutlierDetector
from alibi_detect.models.pytorch.gmm import GMMModel
from alibi_detect.utils.pytorch.misc import get_optimizer


class GMMTorch(TorchOutlierDetector):

    def __init__(
        self,
        n_components: int,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """Pytorch backend for the Gaussian Mixture Model (GMM) outlier detector.

        Parameters
        ----------
        n_components
            Number of components in guassian mixture model.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by
            passing either ``'cuda'``, ``'gpu'`` or ``'cpu'``.
        """
        self.ensembler = None
        self.n_components = n_components
        TorchOutlierDetector.__init__(self, device=device)

    def _fit(
        self,
        x_ref: torch.Tensor,
        optimizer: Callable = torch.optim.Adam,
        learning_rate: float = 0.1,
        batch_size: int = 32,
        epochs: int = 10,
        verbose: int = 0,
    ) -> None:
        """Fit the GMM model.

        Parameters
        ----------
        X
            Training data.
        optimizer
            Optimizer used to train the model.
        learning_rate
            Learning rate used to train the model.
        batch_size
            Batch size used to train the model.
        epochs
            Number of training epochs.
        verbose
            Verbosity level during training. 0 is silent, 1 a progress bar.
        """
        self.model = GMMModel(self.n_components, x_ref.shape[-1]).to(self.device)
        x_ref = x_ref.to(torch.float32)

        batch_size = len(x_ref) if batch_size is None else batch_size
        dataset = TorchDataset(x_ref)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer_instance: torch.optim.Optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.model.train()

        for epoch in range(epochs):
            dl = tqdm(enumerate(dataloader), total=len(dataloader), disable=not verbose)
            loss_ma = 0
            for step, x in dl:
                x = x.to(self.device)
                nll = self.model(x).mean()
                optimizer_instance.zero_grad()
                nll.backward()
                optimizer_instance.step()
                if verbose == 1 and isinstance(dl, tqdm):
                    loss_ma = loss_ma + (nll.item() - loss_ma) / (step + 1)
                    dl.set_description(f'Epoch {epoch + 1}/{epochs}')
                    dl.set_postfix(dict(loss_ma=loss_ma))

    def format_fit_kwargs(self, fit_kwargs: Dict) -> Dict:
        """Format kwargs for `fit` method.

        Parameters
        ----------
        kwargs
            dictionary of Kwargs to format. See `fit` method for details.

        Returns
        -------
        Formatted kwargs.
        """
        return dict(
            optimizer=get_optimizer(fit_kwargs.get('optimizer')),
            learning_rate=fit_kwargs.get('learning_rate', 0.1),
            batch_size=fit_kwargs.get('batch_size', None),
            epochs=(lambda v: 10 if v is None else v)(fit_kwargs.get('epochs', None)),
            verbose=fit_kwargs.get('verbose', 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Detect if `x` is an outlier.

        Parameters
        ----------
        x
            `torch.Tensor` with leading batch dimension.

        Returns
        -------
        `torch.Tensor` of ``bool`` values with leading batch dimension.

        Raises
        ------
        ThresholdNotInferredException
            If called before detector has had `infer_threshold` method called.
        """
        raw_scores = self.score(x)
        scores = self._ensembler(raw_scores)
        if not torch.jit.is_scripting():
            self.check_threshold_inferred()
        preds = scores > self.threshold
        return preds.cpu()

    @torch.no_grad()
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Score `X` using the GMM model.

        Parameters
        ----------
        X
            `torch.Tensor` with leading batch dimension.
        """
        if not torch.jit.is_scripting():
            self.check_fitted()
        X = X.to(torch.float32)
        preds = self.model(X.to(self.device)).cpu()
        return preds
