from copy import deepcopy
from functools import partial
from tqdm import tqdm
import numpy as np
from pykeops.torch import LazyTensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Union, Tuple
from alibi_detect.cd.base import BaseLearnedKernelDrift
from alibi_detect.utils.pytorch import get_device, predict_batch
from alibi_detect.utils.pytorch.data import TorchDataset
from alibi_detect.utils.frameworks import Framework
from alibi_detect.utils._types import TorchDeviceType


class LearnedKernelDriftKeops(BaseLearnedKernelDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            kernel: Union[nn.Module, nn.Sequential],
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            n_permutations: int = 100,
            batch_size_permutations: int = 1000000,
            var_reg: float = 1e-5,
            reg_loss_fn: Callable = (lambda kernel: 0),
            train_size: Optional[float] = .75,
            retrain_from_scratch: bool = True,
            optimizer: torch.optim.Optimizer = torch.optim.Adam,  # type: ignore
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            batch_size_predict: int = 1000000,
            preprocess_batch_fn: Optional[Callable] = None,
            epochs: int = 3,
            num_workers: int = 0,
            verbose: int = 0,
            train_kwargs: Optional[dict] = None,
            device: TorchDeviceType = None,
            dataset: Callable = TorchDataset,
            dataloader: Callable = DataLoader,
            input_shape: Optional[tuple] = None,
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
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before applying the kernel.
        n_permutations
            The number of permutations to use in the permutation test once the MMD has been computed.
        batch_size_permutations
            KeOps computes the n_permutations of the MMD^2 statistics in chunks of batch_size_permutations.
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
        batch_size_predict
            Batch size used for the trained drift detector predictions.
        preprocess_batch_fn
            Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
            processed by the kernel.
        epochs
            Number of training epochs for the kernel. Corresponds to the smaller of the reference and test sets.
        num_workers
            Number of workers for the dataloader. The default (`num_workers=0`) means multi-process data loading
            is disabled. Setting `num_workers>0` may be unreliable on Windows.
        verbose
            Verbosity level during the training of the kernel. 0 is silent, 1 a progress bar.
        train_kwargs
            Optional additional kwargs when training the kernel.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``. Relevant for 'pytorch' and 'keops' backends.
        dataset
            Dataset object used during training.
        dataloader
            Dataloader object used during training. Only relevant for 'pytorch' backend.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            x_ref_preprocessed=x_ref_preprocessed,
            preprocess_at_init=preprocess_at_init,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            n_permutations=n_permutations,
            train_size=train_size,
            retrain_from_scratch=retrain_from_scratch,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta.update({'backend': Framework.KEOPS.value})

        # Set device, define model and training kwargs
        self.device = get_device(device)
        self.original_kernel = kernel
        self.kernel = deepcopy(kernel)

        # Check kernel format
        self.has_proj = hasattr(self.kernel, 'proj') and isinstance(self.kernel.proj, nn.Module)
        self.has_kernel_b = hasattr(self.kernel, 'kernel_b') and isinstance(self.kernel.kernel_b, nn.Module)

        # Define kwargs for dataloader and trainer
        self.dataset = dataset
        self.dataloader = partial(dataloader, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=num_workers)
        self.train_kwargs = {'optimizer': optimizer, 'epochs': epochs,  'preprocess_fn': preprocess_batch_fn,
                             'reg_loss_fn': reg_loss_fn, 'learning_rate': learning_rate, 'verbose': verbose}
        if isinstance(train_kwargs, dict):
            self.train_kwargs.update(train_kwargs)

        self.j_hat = LearnedKernelDriftKeops.JHat(
            self.kernel, var_reg, self.has_proj, self.has_kernel_b).to(self.device)

        # Set prediction and permutation batch sizes
        self.batch_size_predict = batch_size_predict
        self.batch_size_perms = batch_size_permutations
        self.n_batches = 1 + (n_permutations - 1) // batch_size_permutations

    class JHat(nn.Module):
        """
        A module that wraps around the kernel. When passed a batch of reference and batch of test
        instances it returns an estimate of a correlate of test power.
        Equation 4 of https://arxiv.org/abs/2002.09116
        """
        def __init__(self, kernel: nn.Module, var_reg: float, has_proj: bool, has_kernel_b: bool):
            super().__init__()
            self.kernel = kernel
            self.has_proj = has_proj
            self.has_kernel_b = has_kernel_b
            self.var_reg = var_reg

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            n = len(x)
            if self.has_proj and isinstance(self.kernel.proj, nn.Module):
                x_proj, y_proj = self.kernel.proj(x), self.kernel.proj(y)
            else:
                x_proj, y_proj = x, y
            x2_proj, x_proj = LazyTensor(x_proj[None, :, :]), LazyTensor(x_proj[:, None, :])
            y2_proj, y_proj = LazyTensor(y_proj[None, :, :]), LazyTensor(y_proj[:, None, :])
            if self.has_kernel_b:
                x2, x = LazyTensor(x[None, :, :]), LazyTensor(x[:, None, :])
                y2, y = LazyTensor(y[None, :, :]), LazyTensor(y[:, None, :])
            else:
                x, x2, y, y2 = None, None, None, None

            k_xy = self.kernel(x_proj, y2_proj, x, y2)
            k_xx = self.kernel(x_proj, x2_proj, x, x2)
            k_yy = self.kernel(y_proj, y2_proj, y, y2)
            h_mat = k_xx + k_yy - k_xy - k_xy.t()

            h_i = h_mat.sum(1).squeeze(-1)
            h = h_i.sum()
            mmd2_est = (h - n) / (n * (n - 1))
            var_est = 4 * h_i.square().sum() / (n ** 3) - 4 * h.square() / (n ** 4)
            reg_var_est = var_est + self.var_reg

            return mmd2_est/reg_var_est.sqrt()

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, float]:
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
        p-value obtained from the permutation test, the MMD^2 between the reference and test set, \
        and the MMD^2 threshold above which drift is flagged.
        """
        x_ref, x_cur = self.preprocess(x)
        (x_ref_tr, x_cur_tr), (x_ref_te, x_cur_te) = self.get_splits(x_ref, x_cur)
        dl_ref_tr, dl_cur_tr = self.dataloader(self.dataset(x_ref_tr)), self.dataloader(self.dataset(x_cur_tr))

        self.kernel = deepcopy(self.original_kernel) if self.retrain_from_scratch else self.kernel
        self.kernel = self.kernel.to(self.device)
        train_args = [self.j_hat, (dl_ref_tr, dl_cur_tr), self.device]
        LearnedKernelDriftKeops.trainer(*train_args, **self.train_kwargs)  # type: ignore

        m, n = len(x_ref_te), len(x_cur_te)
        if isinstance(x_ref_te, np.ndarray) and isinstance(x_cur_te, np.ndarray):
            x_all = torch.from_numpy(np.concatenate([x_ref_te, x_cur_te], axis=0)).float()
        else:
            x_all = x_ref_te + x_cur_te  # type: ignore[assignment]

        perms = [torch.randperm(m + n) for _ in range(self.n_permutations)]
        mmd2, mmd2_permuted = self._mmd2(x_all, perms, m, n)
        if self.device.type == 'cuda':
            mmd2, mmd2_permuted = mmd2.cpu(), mmd2_permuted.cpu()
        p_val = (mmd2 <= mmd2_permuted).float().mean()

        idx_threshold = int(self.p_val * len(mmd2_permuted))
        distance_threshold = torch.sort(mmd2_permuted, descending=True).values[idx_threshold]
        return p_val.numpy().item(), mmd2.numpy().item(), distance_threshold.numpy()

    def _mmd2(self, x_all: Union[list, torch.Tensor], perms: List[torch.Tensor], m: int, n: int) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched (across the permutations) MMD^2 computation for the original test statistic and the permutations.

        Parameters
        ----------
        x_all
            Concatenated reference and test instances.
        perms
            List with permutation vectors.
        m
            Number of reference instances.
        n
            Number of test instances.

        Returns
        -------
        MMD^2 statistic for the original and permuted reference and test sets.
        """
        preprocess_batch_fn = self.train_kwargs['preprocess_fn']
        if isinstance(preprocess_batch_fn, Callable):  # type: ignore[arg-type]
            x_all = preprocess_batch_fn(x_all)  # type: ignore[operator]
        if self.has_proj:
            x_all_proj = predict_batch(x_all, self.kernel.proj, device=self.device, batch_size=self.batch_size_predict,
                                       dtype=x_all.dtype if isinstance(x_all, torch.Tensor) else torch.float32)
        else:
            x_all_proj = x_all

        x, x2, y, y2 = None, None, None, None
        k_xx, k_yy, k_xy = [], [], []
        for batch in range(self.n_batches):
            i, j = batch * self.batch_size_perms, (batch + 1) * self.batch_size_perms
            # Stack a batch of permuted reference and test tensors and their projections
            x_proj = torch.cat([x_all_proj[perm[:m]][None, :, :] for perm in perms[i:j]], 0)
            y_proj = torch.cat([x_all_proj[perm[m:]][None, :, :] for perm in perms[i:j]], 0)
            if self.has_kernel_b:
                x = torch.cat([x_all[perm[:m]][None, :, :] for perm in perms[i:j]], 0)
                y = torch.cat([x_all[perm[m:]][None, :, :] for perm in perms[i:j]], 0)
            if batch == 0:
                x_proj = torch.cat([x_all_proj[None, :m, :], x_proj], 0)
                y_proj = torch.cat([x_all_proj[None, m:, :], y_proj], 0)
                if self.has_kernel_b:
                    x = torch.cat([x_all[None, :m, :], x], 0)  # type: ignore[call-overload]
                    y = torch.cat([x_all[None, m:, :], y], 0)  # type: ignore[call-overload]
            x_proj, y_proj = x_proj.to(self.device), y_proj.to(self.device)
            if self.has_kernel_b:
                x, y = x.to(self.device), y.to(self.device)

            # Batch-wise kernel matrix computation over the permutations
            with torch.no_grad():
                x2_proj, x_proj = LazyTensor(x_proj[:, None, :, :]), LazyTensor(x_proj[:, :, None, :])
                y2_proj, y_proj = LazyTensor(y_proj[:, None, :, :]), LazyTensor(y_proj[:, :, None, :])
                if self.has_kernel_b:
                    x2, x = LazyTensor(x[:, None, :, :]), LazyTensor(x[:, :, None, :])
                    y2, y = LazyTensor(y[:, None, :, :]), LazyTensor(y[:, :, None, :])
                k_xy.append(self.kernel(x_proj, y2_proj, x, y2).sum(1).sum(1).squeeze(-1))
                k_xx.append(self.kernel(x_proj, x2_proj, x, x2).sum(1).sum(1).squeeze(-1))
                k_yy.append(self.kernel(y_proj, y2_proj, y, y2).sum(1).sum(1).squeeze(-1))

        c_xx, c_yy, c_xy = 1 / (m * (m - 1)), 1 / (n * (n - 1)), 2. / (m * n)
        # Note that the MMD^2 estimates assume that the diagonal of the kernel matrix consists of 1's
        stats = c_xx * (torch.cat(k_xx) - m) + c_yy * (torch.cat(k_yy) - n) - c_xy * torch.cat(k_xy)
        return stats[0], stats[1:]

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
