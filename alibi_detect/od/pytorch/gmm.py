from typing import Callable, Optional, Union
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from alibi_detect.utils.pytorch.data import TorchDataset
from alibi_detect.utils.pytorch.prediction import predict_batch
from alibi_detect.od.pytorch.base import TorchOutlierDetector
from alibi_detect.models.pytorch.gmm import GMMModel


class GMMTorch(TorchOutlierDetector):
    def __init__(
        self,
        n_components: int,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """
        Fits a Gaussian mixture model to the training data and scores new data points
        via the negative log-liklihood under the corresponding density function.
        Parameters
        ----------
        n_components:
            The number of Gaussian mixture components.
        optimizer:
            Used to learn the GMM params.
        rest should be obvious.
        """
        self.n_components = n_components
        TorchOutlierDetector.__init__(self, device=device)

    def _fit(
            self,
            X: torch.Tensor,
            optimizer: Callable = torch.optim.Adam,
            learning_rate: float = 0.1,
            batch_size: int = 32,
            epochs: int = 10,
            verbose: int = 0,
            ) -> None:
        self.model = GMMModel(self.n_components, X.shape[-1]).to(self.device)
        X = X.to(torch.float32)

        dataset = TorchDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.model.train()

        for epoch in range(epochs):
            dl = tqdm(enumerate(dataloader), total=len(dataloader), disable=not verbose)
            loss_ma = 0
            for step, x in dl:
                x = x.to(self.device)
                nll = self.model(x).mean()
                optimizer.zero_grad()  # type: ignore
                nll.backward()
                optimizer.step()  # type: ignore
                if verbose == 1 and isinstance(dl, tqdm):
                    loss_ma = loss_ma + (nll.item() - loss_ma) / (step + 1)
                    dl.set_description(f'Epoch {epoch + 1}/{epochs}')
                    dl.set_postfix(dict(loss_ma=loss_ma))

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

    def score(self, X: torch.Tensor) -> torch.Tensor:
        self.check_fitted()
        batch_size, *_ = X.shape
        X = X.to(torch.float32)
        preds = predict_batch(
            X, self.model.eval(),
            device=self.device,
            batch_size=batch_size
        )
        return torch.tensor(preds)
