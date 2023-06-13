from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from typing_extensions import Literal, Self
from tqdm import tqdm

from alibi_detect.od.pytorch.base import TorchOutlierDetector
from alibi_detect.utils.pytorch.losses import hinge_loss


class SVMTorch(TorchOutlierDetector):
    ensemble = False

    def __init__(
        self,
        n_components: Optional[int] = None,
        device: Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']] = None,
        kernel: Optional[torch.nn.Module] = None,
    ):
        """Pytorch backend for the Support Vector Machine (SVM) outlier detector.

        Parameters
        ----------
        kernel:
            Used to define similarity between data points.
        n_components
            Number of components in the Nystroem approximation By default uses all of them.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by
            passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``.

        Raises
        ------
        ValueError
            If `n_components` is less than 1.
        """
        super().__init__(device=device)
        if n_components < 1:
            raise ValueError('n_components must be at least 1')
        self.n_components = n_components
        self.kernel = kernel
        self.nystroem = Nystroem(kernel, n_components)

    def fit(  # type: ignore[override]
        self,
        x_ref: torch.Tensor,
        nu: float,
        step_size_range: Tuple[float, float] = (1e-6, 1.0),
        n_step_sizes: int = 16,
        tol: float = 1e-6,
        n_iter_no_change: int = 25,
        max_iter: int = 1000,
        verbose: int = 0,
    ) -> Dict:
        """Fit the SVM detector.

        Parameters
        ----------
        x_ref
            Training data.
        nu:
            The proportion of the training data that should be considered outliers. Note that this does
            not necessarily correspond to the false positive rate on test data, which is still defined when
            calling the `infer_threshold()` method.
        step_size_range:
            The range of values to be considered for the gradient descent step size at each iteration.
        n_step_sizes:
            The number of step sizes in the defined range to be tested for loss reduction. This many points
            are spaced equidistantly along the range in log space.
        tol:
            The decrease in loss required over the previous n_iter_no_change iterations in order to
            continue optimizing.
        n_iter_no_change:
            The number of iterations over which the loss must decrease by `tol` in order for
            optimization to continue.
        max_iter:
            The maximum number of optimization steps.
        verbose
            Verbosity level during training. 0 is silent, 1 a progress bar.

        Returns
        -------
        Dictionary with fit results. The dictionary contains the following keys:
            - converged: bool indicating whether EM algorithm converged.
            - n_iter: number of EM iterations performed.
            - lower_bound: log-likelihood lower bound.
        """
        X_nys = self.nystroem.fit(x_ref).transform(x_ref)
        n, d = X_nys.shape
        min_eta, max_eta = step_size_range
        etas = torch.tensor(
            np.linspace(
                np.log(min_eta),
                np.log(max_eta),
                n_step_sizes
            ),
            dtype=X_nys.dtype,
            device=self.device
        ).exp()

        # Initialise coeffs/preds/loss
        coeffs = torch.zeros(d, dtype=X_nys.dtype, device=self.device)
        intercept = torch.zeros(1, dtype=X_nys.dtype, device=self.device)
        preds = X_nys @ coeffs + intercept
        loss = nu * (coeffs.square().sum()/2 + intercept) + hinge_loss(preds)
        min_loss, min_loss_coeffs, min_loss_intercept = loss, coeffs, intercept
        iter, t_since_improv = 0, 0
        converged = False

        with tqdm(total=max_iter, disable=not verbose) as pbar:
            while not converged:
                pbar.update(1)
                # First two lines give form of sgd update (for each candidate step size)
                sup_vec_inds = (preds < 1)
                cand_coeffs = coeffs[:, None] * \
                    (1-etas*nu) + etas*(X_nys[sup_vec_inds].sum(0)/n)[:, None]
                cand_intercept = intercept - etas*nu + (sup_vec_inds.sum()/n)

                # Compute loss for each candidate step size and choose the best
                cand_preds = X_nys @ cand_coeffs + cand_intercept
                cand_losses = nu * (cand_coeffs.square().sum(0)/2 + cand_intercept) + hinge_loss(cand_preds)
                best_step_size = cand_losses.argmin()
                coeffs, intercept = cand_coeffs[:, best_step_size], cand_intercept[best_step_size]
                preds, loss = cand_preds[:, best_step_size], cand_losses[best_step_size]

                # Keep track of best performing coefficients and time since improving (by more than tol)
                if loss < min_loss:
                    if loss < min_loss - tol:
                        t_since_improv = 0
                    min_loss, min_loss_coeffs, min_loss_intercept = loss, coeffs, intercept
                else:
                    t_since_improv += 1

                # Decide whether to continue
                if iter > max_iter or t_since_improv > n_iter_no_change:
                    self.coeffs = min_loss_coeffs
                    self.intercept = min_loss_intercept
                    converged = True
                    break
                else:
                    iter += 1

                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix(dict(loss=loss.cpu().detach().numpy().item()))

        self._set_fitted()
        return {
            'converged': converged,
            'lower_bound': min_loss,
            'n_iter': iter
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
            nu=fit_kwargs.get('nu', None),
            step_size_range=fit_kwargs.get('step_size_range', (1e-6, 1.0)),
            n_iter_no_change=fit_kwargs.get('n_iter_no_change', 25),
            tol=fit_kwargs.get('tol', 1e-6),
            verbose=fit_kwargs.get('verbose', 0),
            n_step_sizes=fit_kwargs.get('n_step_sizes', 16),
            max_iter=fit_kwargs.get('max_iter', 1000)
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
        x_nys = self.nystroem.transform(x)
        preds = x_nys @ self.coeffs + self.intercept
        return -preds


class Nystroem:
    def __init__(
        self,
        kernel: Callable,
        n_components: Optional[int] = None
    ) -> None:
        """Nystroem Approximation of a kernel.

        Parameters
        ----------
        kernel
            Kernel function.
        n_components, optional
            Number of components in the Nystroem approximation. By default uses all of them.
        """
        self.kernel = kernel
        self.n_components = n_components

    def fit(
        self,
        x: torch.Tensor
    ) -> Self:
        """Fit the Nystroem approximation.

        Parameters
        ----------
        x
            `torch.Tensor` of shape ``(n, d)`` where ``n`` is the number of samples and ``d`` is the dimensionality of
            the data.
        """
        n = len(x)
        n_components = n if self.n_components is None else self.n_components
        inds = torch.randperm(n)[:n_components]
        self.z = x[inds]
        K_zz = self.kernel(self.z,  self.z)
        K_zz += 1e-16 + torch.eye(n_components, device=K_zz.device)
        U, S, V = torch.linalg.svd(K_zz)
        self.K_zz_root_inv = (U / S.sqrt()) @ V
        return self

    def transform(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Transform `x` into the Nystroem approximation.

        Parameters
        ----------
        x
            `torch.Tensor` of shape ``(n, d)`` where ``n`` is the number of samples and ``d`` is the dimensionality of
            the data.

        Returns
        -------
        `torch.Tensor` of shape ``(n, n_components)`` where ``n_components`` is the number of components in the
        Nystroem approximation.
        """

        K_xz = self.kernel(x, self.z)
        return K_xz @ self.K_zz_root_inv
