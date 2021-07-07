from __future__ import annotations
from copy import deepcopy
from tqdm import tqdm
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
from typing import Callable, Dict, Optional, Union, Tuple
from alibi_detect.cd.base import BaseNMEDrift

logger = logging.getLogger(__name__)


class NMEDriftTorch(BaseNMEDrift):
    def __init__(
            self,
            x_ref: np.ndarray,
            kernel: Union[nn.Module, nn.Sequential],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            train_size: Optional[float] = .75,
            cov_reg: float = 1e-6,
            optimizer: Callable = torch.optim.Adam,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 3,
            verbose: int = 0,
            train_kwargs: Optional[dict] = None,
            device: Optional[str] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Classifier-based drift detector. The classifier is trained on a fraction of the combined
        reference and test data and drift is detected on the remaining data. To use all the data
        to detect drift, a stratified cross-validation scheme can be chosen.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        model
            PyTorch classification model used for drift detection.
        p_val
            p-value used for the significance of the test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        preds_type
            Whether the model outputs 'probs' or 'logits'
        binarize_preds
            Whether to test for discrepency on soft (e.g. probs/logits) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the classifier.
            The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`.
        n_folds
            Optional number of stratified folds used for training. The model preds are then calculated
            on all the out-of-fold predictions. This allows to leverage all the reference and test data
            for drift detection at the expense of longer computation. If both `train_size` and `n_folds`
            are specified, `n_folds` is prioritized.
        seed
            Optional random seed for fold selection.
        optimizer
            Optimizer used during training of the classifier.
        learning_rate
            Learning rate used by optimizer.
        batch_size
            Batch size used during training of the classifier.
        epochs
            Number of training epochs for the classifier for each (optional) fold.
        verbose
            Verbosity level during the training of the classifier. 0 is silent, 1 a progress bar.
        train_kwargs
            Optional additional kwargs when fitting the classifier.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            train_size=train_size,
            cov_reg=cov_reg,
            data_type=data_type
        )
        self.meta.update({'backend': 'pytorch'})

        # set device, define model and training kwargs
        if device is None or device.lower() in ['gpu', 'cuda']:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cpu':
                logger.warning('No GPU detected, fall back on CPU.')
        else:
            self.device = torch.device('cpu')

        self.kernel = kernel

        # define kwargs for dataloader and trainer
        self.dl_kwargs = {'batch_size': batch_size, 'shuffle': True}
        self.train_kwargs = {'optimizer': optimizer, 'epochs': epochs,
                             'learning_rate': learning_rate, 'verbose': verbose}
        if isinstance(train_kwargs, dict):
            self.train_kwargs.update(train_kwargs)

    class NMEEmbedder(nn.Module):
        def __init__(self, kernel: nn.Module, init_locations: torch.Tensor):
            super().__init__()
            self.kernel = kernel
            self.test_locs = nn.Parameter(init_locations)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            k_xtl = self.kernel(x, self.test_locs)
            k_ytl = self.kernel(y, self.test_locs)
            return k_xtl - k_ytl

    def score(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Compute the out-of-fold drift metric such as the accuracy from a classifier
        trained to distinguish the reference data from the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value, and a notion of distance between the trained classifier's out-of-fold performance
        and that which we'd expect under the null assumption of no drift.
        """
        x_ref, x = self.preprocess(x)

        init_test_locations = torch.as_tensor(self.init_test_locations(x_ref))
        (x_ref_train, x_ref_test), (x_train, x_test) = self.get_splits(x_ref, x)

        ds_tr = TensorDataset(torch.from_numpy(x_ref_train), torch.from_numpy(x_train))
        dl_tr = DataLoader(ds_tr, **self.dl_kwargs)  # type: ignore
        nme_embedder = NMEDriftTorch.NMEEmbedder(self.kernel, init_test_locations)
        train_args = [nme_embedder, self.cov_reg, dl_tr, self.device]
        self.trainer(*train_args, **self.train_kwargs)  # type: ignore

        nme_embedder.eval()
        embeddings = NMEDriftTorch.embed_batch(
            torch.as_tensor(x_ref_test), torch.as_tensor(x_test), nme_embedder, self.device, self.dl_kwargs['batch_size']
        )
        nme_estimate = NMEDriftTorch.embedding_to_estimate(embeddings, cov_reg=self.cov_reg)
        new_test_locs = nme_embedder.test_locs.detach().cpu().numpy()

        p_val = stats.chi2.sf(nme_estimate.cpu().numpy(), self.J)
        return p_val, nme_estimate, new_test_locs

    @staticmethod
    def embedding_to_estimate(z: torch.Tensor, cov_reg: float = 1e-12) -> torch.Tensor:
        n, J = z.shape
        S = torch.einsum('ij,ik->jk', (z - z.mean(0)), (z - z.mean(0)))/(n-1)
        S += cov_reg * torch.eye(J)
        S_inv = torch.inverse(S)
        return n * z.mean(0).reshape(1, J) @ S_inv @ z.mean(0).reshape(J, 1)

    @staticmethod
    def embed_batch(
        x_ref: torch.Tensor,
        x: torch.Tensor,
        nme_embedder: NMEEmbedder,
        device: torch.device,
        batch_size: int
    ) -> torch.Tensor:
        n = len(x)
        n_minibatch = int(np.ceil(n / batch_size))
        embeddings = []
        with torch.no_grad():
            for i in range(n_minibatch):
                istart, istop = i * batch_size, min((i + 1) * batch_size, n)
                x_ref_batch, x_batch = x_ref[istart:istop], x[istart:istop]
                embeddings_batch = nme_embedder(x_ref_batch.to(device), x_batch.to(device))
                embeddings.append(embeddings_batch)
        return torch.cat(embeddings, 0)

    @staticmethod
    def trainer(
        nme_embedder: NMEEmbedder,
        cov_reg: float,
        dataloader: DataLoader,
        device: torch.device,
        optimizer: Callable = torch.optim.Adam,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        verbose: int = 1,
    ) -> None:
        optimizer = optimizer(nme_embedder.parameters(), lr=learning_rate)
        nme_embedder.train()
        est_ma = 0.
        for epoch in range(epochs):
            dl = tqdm(enumerate(dataloader), total=len(dataloader)) if verbose == 1 else enumerate(dataloader)
            for step, (x_ref, x) in dl:
                x_ref, x = x_ref.to(device), x.to(device)
                embedding = nme_embedder(x_ref, x)
                optimizer.zero_grad()  # type: ignore
                estimate = NMEDriftTorch.embedding_to_estimate(embedding, cov_reg=cov_reg)
                (-estimate).backward()  # ascent
                optimizer.step()  # type: ignore
                if verbose == 1:
                    est_ma = est_ma + (estimate.item() - est_ma) / (step + 1)
                    dl.set_description(f'Epoch {epoch + 1}/{epochs}')
                    dl.set_postfix(dict(nme_estimate=est_ma))
