from typing import Dict, Type
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from alibi_detect.utils.pytorch.data import TorchDataset
from alibi_detect.od.pytorch.base import TorchOutlierDetector
from alibi_detect.models.pytorch.gmm import GMMModel
from alibi_detect.utils.pytorch.misc import get_optimizer
from alibi_detect.utils._types import TorchDeviceType


class GMMTorch(TorchOutlierDetector):
    ensemble = False

    def __init__(
        self,
        n_components: int,
        device: TorchDeviceType = None,
    ):
        """Pytorch backend for the Gaussian Mixture Model (GMM) outlier detector.

        Parameters
        ----------
        n_components
            Number of components in gaussian mixture model.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.

        Raises
        ------
        ValueError
            If `n_components` is less than 1.
        """
        super().__init__(device=device)
        if n_components < 1:
            raise ValueError('n_components must be at least 1')
        self.n_components = n_components

    def fit(  # type: ignore[override]
        self,
        x_ref: torch.Tensor,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        learning_rate: float = 0.1,
        max_epochs: int = 10,
        batch_size: int = 32,
        tol: float = 1e-3,
        n_iter_no_change: int = 25,
        verbose: int = 0,
    ) -> Dict:
        """Fit the GMM model.

        Parameters
        ----------
        x_ref
            Training data.
        optimizer
            Optimizer used to train the model.
        learning_rate
            Learning rate used to train the model.
        max_epochs
            Maximum number of training epochs.
        batch_size
            Batch size used to train the model.
        tol
            Convergence threshold. Training iterations will stop when the lower bound average
            gain is below this threshold.
        n_iter_no_change
            The number of iterations over which the loss must decrease by `tol` in order for
            optimization to continue.
        verbose
            Verbosity level during training. 0 is silent, 1 a progress bar.

        Returns
        -------
        Dictionary with fit results. The dictionary contains the following keys:
            - converged: bool indicating whether training converged.
            - n_epochs: number of gradient descent iterations performed.
            - lower_bound: log-likelihood lower bound.
        """
        self.model = GMMModel(self.n_components, x_ref.shape[-1]).to(self.device)
        x_ref = x_ref.to(torch.float32)

        batch_size = len(x_ref) if batch_size is None else batch_size
        dataset = TorchDataset(x_ref)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        optimizer_instance: torch.optim.Optimizer = optimizer(  # type: ignore[call-arg]
            self.model.parameters(),
            lr=learning_rate
        )
        self.model.train()

        min_loss = None
        converged = False
        epoch = 0

        while not converged and epoch < max_epochs:
            epoch += 1
            dl = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                disable=not verbose
            )
            loss_ma = 0
            for step, x in dl:
                x = x.to(self.device)
                nll = self.model(x).mean()
                optimizer_instance.zero_grad()
                nll.backward()
                optimizer_instance.step()

                if verbose and isinstance(dl, tqdm):
                    loss_ma = loss_ma + (nll.item() - loss_ma) / (step + 1)
                    dl.set_description(f'Epoch {epoch + 1}/{max_epochs}')
                    dl.set_postfix(dict(loss_ma=loss_ma))

                if min_loss is None or nll < min_loss - tol:
                    t_since_improv = 0
                    min_loss = nll
                else:
                    t_since_improv += 1

                if t_since_improv > n_iter_no_change:
                    converged = True
                    break

        self._set_fitted()
        return {
            'converged': converged,
            'lower_bound': self._to_frontend_dtype(min_loss),
            'n_epochs': epoch
        }

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
            max_epochs=(lambda v: 10 if v is None else v)(fit_kwargs.get('max_epochs', None)),
            verbose=fit_kwargs.get('verbose', 0),
            tol=fit_kwargs.get('tol', 1e-3),
            n_iter_no_change=fit_kwargs.get('n_iter_no_change', 25)
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
        scores = self.score(x)
        if not torch.jit.is_scripting():
            self.check_threshold_inferred()
        preds = scores > self.threshold
        return preds

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the score of `x`

        Parameters
        ----------
        x
            `torch.Tensor` with leading batch dimension.

        Returns
        -------
        `torch.Tensor` of scores with leading batch dimension.

        Raises
        ------
        NotFittedError
            Raised if method called and detector has not been fit.
        """
        if not torch.jit.is_scripting():
            self.check_fitted()
        x = x.to(torch.float32)
        preds = self.model(x.to(self.device))
        return preds
