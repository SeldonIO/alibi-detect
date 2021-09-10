from copy import deepcopy
from functools import partial
from tqdm import tqdm
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Dict, Optional, Union, Tuple
from alibi_detect.cd.base import BaseLearnedKernelDrift
from alibi_detect.utils.pytorch.distance import mmd2_from_kernel_matrix, batch_compute_kernel_matrix
from alibi_detect.utils.pytorch.data import TorchDataset

logger = logging.getLogger(__name__)


class LearnedKernelDriftTorch(BaseLearnedKernelDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            kernel: Union[nn.Module, nn.Sequential],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            n_permutations: int = 100,
            var_reg: float = 1e-5,
            reg_loss_fn: Callable = (lambda kernel: 0),
            train_size: Optional[float] = .75,
            retrain_from_scratch: bool = True,
            optimizer: torch.optim.Optimizer = torch.optim.Adam,  # type: ignore
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            preprocess_batch_fn: Optional[Callable] = None,
            epochs: int = 3,
            verbose: int = 0,
            train_kwargs: Optional[dict] = None,
            device: Optional[str] = None,
            dataset: Callable = TorchDataset,
            dataloader: Callable = DataLoader,
            data_type: Optional[str] = None
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) data drift detector where the kernel is trained to maximise an
        estimate of the test power. The kernel is trained on a split of the reference and test instances
        and then the MMD is evaluated on held out instances and a permutation test is performed.

        For details see Liu et al (2020): Learning Deep Kernels for Non-Parametric Two-Sample Tests
        (https://arxiv.org/abs/2002.09116)


        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        kernel
            Trainable PyTorch module that returns a similarity between two instances.
        p_val
            p-value used for the significance of the test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before applying the kernel.
        n_permutations
            The number of permutations to use in the permutation test once the MMD has been computed.
        var_reg
            Constant added to the estimated variance of the MMD for stability.
        reg_loss_fn
            The regularisation term reg_loss_fn(kernel) is added to the loss function being optimized.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the kernel.
            The drift is detected on `1 - train_size`.
        retrain_from_scratch
            Whether the kernel should be retrained from scratch for each set of test data or whether
            it should instead continue training from where it left off on the previous set.
        optimizer
            Optimizer used during training of the kernel.
        learning_rate
            Learning rate used by optimizer.
        batch_size
            Batch size used during training of the kernel.
        preprocess_batch_fn
            Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
            processed by the kernel.
        epochs
            Number of training epochs for the kernel. Corresponds to the smaller of the reference and test sets.
        verbose
            Verbosity level during the training of the kernel. 0 is silent, 1 a progress bar.
        train_kwargs
            Optional additional kwargs when training the kernel.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
        dataset
            Dataset object used during training.
        dataloader
            Dataloader object used during training. Only relevant for 'pytorch' backend.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            n_permutations=n_permutations,
            train_size=train_size,
            retrain_from_scratch=retrain_from_scratch,
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
        self.original_kernel = kernel
        self.kernel = deepcopy(kernel)

        # define kwargs for dataloader and trainer
        self.dataset = dataset
        self.dataloader = partial(dataloader, batch_size=batch_size, shuffle=True, drop_last=True)
        self.kernel_mat_fn = partial(
            batch_compute_kernel_matrix, device=self.device, preprocess_fn=preprocess_batch_fn, batch_size=batch_size
        )
        self.train_kwargs = {'optimizer': optimizer, 'epochs': epochs,  'preprocess_fn': preprocess_batch_fn,
                             'reg_loss_fn': reg_loss_fn, 'learning_rate': learning_rate, 'verbose': verbose}
        if isinstance(train_kwargs, dict):
            self.train_kwargs.update(train_kwargs)

        self.j_hat = LearnedKernelDriftTorch.JHat(self.kernel, var_reg).to(self.device)

    class JHat(nn.Module):
        """
        A module that wraps around the kernel. When passed a batch of reference and batch of test
        instances it returns an estimate of a correlate of test power.
        Equation 4 of https://arxiv.org/abs/2002.09116
        """
        def __init__(self, kernel: nn.Module, var_reg: float):
            super().__init__()
            self.kernel = kernel
            self.var_reg = var_reg

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            k_xx, k_yy, k_xy = self.kernel(x, x), self.kernel(y, y), self.kernel(x, y)
            h_mat = k_xx + k_yy - k_xy - k_xy.t()

            n = len(x)
            mmd2_est = (h_mat.sum()-h_mat.trace())/(n*(n-1))
            var_est = 4*h_mat.sum(-1).square().sum()/(n**3) - 4*h_mat.sum().square()/(n**4)
            reg_var_est = var_est + self.var_reg

            return mmd2_est/reg_var_est.sqrt()

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, np.ndarray]:
        """
        Compute the p-value resulting from a permutation test using the maximum mean discrepancy
        as a distance measure between the reference data and the data to be tested. The kernel
        used within the MMD is first trained to maximise an estimate of the resulting test power.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the MMD^2 between the reference and test set
        and the MMD^2 values from the permutation test.
        """
        x_ref, x_cur = self.preprocess(x)
        (x_ref_tr, x_cur_tr), (x_ref_te, x_cur_te) = self.get_splits(x_ref, x_cur)
        dl_ref_tr, dl_cur_tr = self.dataloader(self.dataset(x_ref_tr)), self.dataloader(self.dataset(x_cur_tr))

        self.kernel = deepcopy(self.original_kernel) if self.retrain_from_scratch else self.kernel
        self.kernel = self.kernel.to(self.device)
        train_args = [self.j_hat, (dl_ref_tr, dl_cur_tr), self.device]
        LearnedKernelDriftTorch.trainer(*train_args, **self.train_kwargs)  # type: ignore

        if isinstance(x_ref_te, np.ndarray) and isinstance(x_cur_te, np.ndarray):
            x_all = np.concatenate([x_ref_te, x_cur_te], axis=0)
        else:
            x_all = x_ref_te + x_cur_te
        kernel_mat = self.kernel_mat_fn(x_all, x_all, self.kernel)
        kernel_mat = kernel_mat - torch.diag(kernel_mat.diag())  # zero diagonal
        mmd2 = mmd2_from_kernel_matrix(kernel_mat, len(x_cur_te), permute=False, zero_diag=False)
        mmd2_permuted = torch.Tensor(
            [mmd2_from_kernel_matrix(kernel_mat, len(x_cur_te), permute=True, zero_diag=False)
                for _ in range(self.n_permutations)]
        )
        if self.device.type == 'cuda':
            mmd2, mmd2_permuted = mmd2.cpu(), mmd2_permuted.cpu()

        p_val = (mmd2 <= mmd2_permuted).float().mean()
        return p_val.numpy().item(), mmd2.numpy().item(), mmd2_permuted.numpy()

    @staticmethod
    def trainer(
        j_hat: JHat,
        dataloaders: Tuple[DataLoader, DataLoader],
        device: torch.device,
        optimizer: Callable = torch.optim.Adam,
        learning_rate: float = 1e-3,
        preprocess_fn: Callable = None,
        epochs: int = 20,
        reg_loss_fn: Callable = (lambda kernel: 0),
        verbose: int = 1,
    ) -> None:
        """
        Train the kernel to maximise an estimate of test power using minibatch gradient descent.
        """
        optimizer = optimizer(j_hat.parameters(), lr=learning_rate)
        j_hat.train()
        loss_ma = 0.
        for epoch in range(epochs):
            dl_ref, dl_cur = dataloaders
            dl = tqdm(enumerate(zip(dl_ref, dl_cur)), total=min(len(dl_ref), len(dl_cur))) if verbose == 1 else \
                enumerate(zip(dl_ref, dl_cur))
            for step, (x_ref, x_cur) in dl:
                if isinstance(preprocess_fn, Callable):  # type: ignore
                    x_ref, x_cur = preprocess_fn(x_ref), preprocess_fn(x_cur)
                x_ref, x_cur = x_ref.to(device), x_cur.to(device)
                optimizer.zero_grad()  # type: ignore
                estimate = j_hat(x_ref, x_cur)
                loss = -estimate + reg_loss_fn(j_hat.kernel)  # ascent
                loss.backward()
                optimizer.step()  # type: ignore
                if verbose == 1:
                    loss_ma = loss_ma + (loss.item() - loss_ma) / (step + 1)
                    dl.set_description(f'Epoch {epoch + 1}/{epochs}')
                    dl.set_postfix(dict(loss=loss_ma))
