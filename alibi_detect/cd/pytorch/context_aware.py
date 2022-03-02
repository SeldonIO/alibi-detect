import logging
import numpy as np
import torch
from typing import Callable, Dict, Optional, Tuple, Union, Type
from alibi_detect.cd.base import BaseContextAwareDrift
from alibi_detect.cd.domain_clf import DomainClf, SVCDomainClf
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ContextAwareDriftTorch(BaseContextAwareDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            c_ref: np.ndarray,
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            x_kernel: Callable = None,
            c_kernel: Callable = None,
            domain_clf: Union[Type[DomainClf], DomainClf] = SVCDomainClf,
            n_permutations: int = 1000,
            cond_prop: float = 0.25,
            lams: Optional[Tuple[float, float]] = None,
            batch_size: Optional[int] = 256,
            device: Optional[str] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
            verbose: bool = False
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) based context aware drift detector.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        c_ref
            Data used as context for the reference distribution.
        p_val
            p-value used for the significance of the permutation test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_kernel
            Kernel defined on the input data, defaults to Gaussian RBF kernel.
        c_kernel
            Kernel defined on the context data, defaults to Gaussian RBF kernel.
        domain_clf
            Domain classifier, takes conditioning variables and their domain, and returns propensity scores (probs of
            being test instances). Must be a subclass of :py:class:`~alibi_detect.cd.domain_clf.DomainClf`.
        n_permutations
            Number of permutations used in the permutation test.
        cond_prop
            Proportion of contexts held out to condition on.
        lams
            Ref and test regularisation parameters. Tuned if None.
        batch_size
            If not None, then compute batches of MMDs at a time (rather than all at once).
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        verbose
            Whether or not to print progress during configuration.
        """
        super().__init__(
            x_ref=x_ref,
            c_ref=c_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            x_kernel=x_kernel,
            c_kernel=c_kernel,
            domain_clf=domain_clf,
            n_permutations=n_permutations,
            cond_prop=cond_prop,
            lams=lams,
            batch_size=batch_size,
            input_shape=input_shape,
            data_type=data_type,
            verbose=verbose
        )

        self.meta.update({'backend': 'pytorch'})

        # set backend
        if device is None or device.lower() in ['gpu', 'cuda']:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cpu':
                print('No GPU detected, fall back on CPU.')
        else:
            self.device = torch.device('cpu')

    def score(self,  # type: ignore[override]
              x: Union[np.ndarray, list], c: np.ndarray) -> Tuple[float, float, np.ndarray, Tuple]:
        """
        Compute the MMD based conditional test statistic, and perform a conditional permutation test to obtain a
        p-value representing the test statistic's extremity under the null hypothesis.

        Parameters
        ----------
        x
            Batch of instances.
        c
            Context associated with batch of instances.

        Returns
        -------
        p-value obtained from the conditional permutation test, the MMD-ADiTT test statistic, the permuted
        test statistics, and a tuple containing the coupling matrices (xx, yy, xy).
        """
        x_ref, x = self.preprocess(x)

        # Hold out a portion of contexts for conditioning on
        n_held = int(len(c)*self.cond_prop)
        n_test = len(c) - n_held
        c, c_held = torch.split(torch.as_tensor(c), [n_test, n_held])
        x, _ = torch.split(torch.as_tensor(x), [n_test, n_held])
        n_ref, n_test = self.n, len(x)

        # Combine ref and test data
        x_all = np.concatenate([self.x_ref, x], axis=0)
        c_all = np.concatenate([self.c_ref, c], axis=0)

        # Obtain n_permutations conditional reassignments
        bools = torch.cat([torch.zeros(n_ref), torch.ones(n_test)])
        prop_scores_np = self.clf(c_all, bools.numpy())
        prop_scores = torch.as_tensor(prop_scores_np)
        self.redrawn_bools = [torch.bernoulli(prop_scores) for _ in range(self.n_permutations)]
        iters = tqdm(self.redrawn_bools, total=self.n_permutations) if self.verbose else self.redrawn_bools

        # Compute kernel matrices
        x_all = torch.as_tensor(x_all).to(self.device)
        c_all = torch.as_tensor(c_all).to(self.device)
        K = self.x_kernel(x_all, x_all)
        L = self.c_kernel(c_all, c_all)
        L_held = self.c_kernel(c_held.to(self.device), c_all)

        # Compute test stat on original and reassigned data
        stat, coupling_xx, coupling_yy, coupling_xy = self._cmmd(K, L, bools.to(self.device), L_held=L_held)
        permuted_stats = torch.stack([self._cmmd(K, L, perm_bools, L_held=L_held)[0] for perm_bools in iters])

        # Compute p-value
        p_val = (stat <= permuted_stats).float().mean()
        coupling = (coupling_xx.numpy(), coupling_yy.numpy(), coupling_xy.numpy())

        return p_val.numpy().item(), stat.numpy().item(), permuted_stats.numpy(), coupling

    def _cmmd(self, K: torch.Tensor, L: torch.Tensor, bools: torch.Tensor, L_held: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Private method to compute the MMD-ADiTT test statistic.
        """
        # Form kernel matrices
        n_ref, n_test = (bools == 0).sum(), (bools == 1).sum()
        L_0, L_1 = L[bools == 0][:, bools == 0], L[bools == 1][:, bools == 1]
        K_0, K_1 = K[bools == 0][:, bools == 0], K[bools == 1][:, bools == 1]

        # Initialise regularisation parameters
        # Implemented only for first _cmmd call which corresponds to original window assignment
        if self.lams is None:
            possible_lams = torch.tensor([2**(-i) for i in range(20)]).to(K.device)  # fairly arbitrary atm # TODO
            lam_0 = self._pick_lam(possible_lams, K_0, L_0, n_folds=5)
            lam_1 = self._pick_lam(possible_lams, K_1, L_1, n_folds=5)
            self.lams = (lam_0, lam_1)  # type: ignore[assignment]
            # Ignore above as self.lams is Optional[Tuple[float, float]] in base class.

        # Compute stat
        L_0_inv = torch.linalg.inv(L_0 + n_ref*self.lams[0]*torch.eye(int(n_ref)).to(L_0.device))
        L_1_inv = torch.linalg.inv(L_1 + n_test*self.lams[1]*torch.eye(int(n_test)).to(L_1.device))
        A_0 = L_held[:, bools == 0] @ L_0_inv
        A_1 = L_held[:, bools == 1] @ L_1_inv
        # Allow batches of MMDs to be computed at a time (rather than all)
        if self.batch_size is not None:
            bs = self.batch_size
            coupling_xx = torch.stack([torch.einsum('ij,ik->ijk', A_0_i, A_0_i).mean(0)
                                       for A_0_i in A_0.split(bs)]).mean(0)
            coupling_yy = torch.stack([torch.einsum('ij,ik->ijk', A_1_i, A_1_i).mean(0)
                                       for A_1_i in A_1.split(bs)]).mean(0)
            coupling_xy = torch.stack([
                torch.einsum('ij,ik->ijk', A_0_i, A_1_i).mean(0) for A_0_i, A_1_i in zip(A_0.split(bs), A_1.split(bs))
            ]).mean(0)
        else:
            coupling_xx = torch.einsum('ij,ik->ijk', A_0, A_0).mean(0)
            coupling_yy = torch.einsum('ij,ik->ijk', A_1, A_1).mean(0)
            coupling_xy = torch.einsum('ij,ik->ijk', A_0, A_1).mean(0)
        sim_xx = (K[bools == 0][:, bools == 0]*coupling_xx).sum()
        sim_yy = (K[bools == 1][:, bools == 1]*coupling_yy).sum()
        sim_xy = (K[bools == 0][:, bools == 1]*coupling_xy).sum()
        stat = sim_xx + sim_yy - 2*sim_xy

        return stat.cpu(), coupling_xx.cpu(), coupling_yy.cpu(), coupling_xy.cpu()

    def _pick_lam(self, lams: torch.Tensor, K: torch.Tensor, L: torch.Tensor, n_folds: int = 5) -> torch.Tensor:
        """
        The conditional mean embedding is estimated as the solution of a regularised regression problem.

        This private method function uses cross validation to select the regularisation parameter that
        minimises squared error on the out-of-fold instances. The error is a distance in the RKHS and is
        therefore an MMD-like quantity itself.
        """
        n = len(L)
        fold_size = n // n_folds
        losses = torch.zeros_like(lams, dtype=torch.float).to(K.device)
        for fold in range(n_folds):
            inds_oof = np.arange(n)[(fold*fold_size):((fold+1)*fold_size)]
            inds_if = np.setdiff1d(np.arange(n), inds_oof)
            K_if, L_if = K[inds_if][:, inds_if], L[inds_if][:, inds_if]
            n_if = len(K_if)
            L_inv_lams = torch.stack(
                [torch.linalg.inv(L_if + n_if*lam*torch.eye(n_if).to(L.device)) for lam in lams])  # n_lam x n_if x n_if
            KW = torch.einsum('ij,ljk->lik', K_if, L_inv_lams)
            lW = torch.einsum('ij,ljk->lik', L[inds_oof][:, inds_if], L_inv_lams)
            lWKW = torch.einsum('lij,ljk->lik', lW, KW)
            lWKWl = torch.einsum('lkj,jk->lk', lWKW, L[inds_if][:, inds_oof])  # n_lam x n_oof
            lWk = torch.einsum('lij,ji->li', lW, K[inds_if][:, inds_oof])  # n_lam x n_oof
            kxx = torch.ones_like(lWk).to(lWk.device) * torch.max(K)
            losses += (lWKWl + kxx - 2*lWk).sum(-1)
        return lams[torch.argmin(losses)]
