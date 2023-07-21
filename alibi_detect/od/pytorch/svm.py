import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import SGDOneClassSVM
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm
from typing_extensions import Self

from alibi_detect.od.pytorch.base import TorchOutlierDetector
from alibi_detect.utils.pytorch.losses import hinge_loss
from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.utils._types import TorchDeviceType


class SVMTorch(TorchOutlierDetector):
    ensemble = False

    def __init__(
        self,
        nu: float,
        kernel: 'torch.nn.Module' = None,
        n_components: Optional[int] = None,
        device: TorchDeviceType = None,
    ):
        """Pytorch backend for the Support Vector Machine (SVM) outlier detector.

        Parameters
        ----------
        nu
            The proportion of the training data that should be considered outliers. Note that this does
            not necessarily correspond to the false positive rate on test data, which is still defined when
            calling the `infer_threshold` method.
        kernel
            Kernel function to use for outlier detection.
        n_components
            Number of components in the Nystroem approximation, by default uses all of them.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.
        """
        super().__init__(device=device)
        self.n_components = n_components
        if kernel is None:
            kernel = GaussianRBF()
        self.kernel = kernel
        self.nystroem = _Nystroem(
            self.kernel,
            self.n_components
        )
        self.nu = nu

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


class SgdSVMTorch(SVMTorch):
    ensemble = False

    def __init__(
        self,
        nu: float,
        kernel: 'torch.nn.Module' = None,
        n_components: Optional[int] = None,
        device: TorchDeviceType = None,
    ):
        """SGD Optimization backend for the One class support vector machine (SVM) outlier detector.

        Parameters
        ----------
        nu
            The proportion of the training data that should be considered outliers. Note that this does
            not necessarily correspond to the false positive rate on test data, which is still defined when
            calling the `infer_threshold` method.
        kernel
            Kernel function to use for outlier detection.
        n_components
            Number of components in the Nystroem approximation, by default uses all of them.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.
        """
        if (isinstance(device, str) and device in ('gpu', 'cuda')) or \
                (isinstance(device, torch.device) and device.type == 'cuda'):
            warnings.warn(('If using the `sgd` optimization option with GPU then only the Nystroem approximation'
                           ' portion of the method will utilize the GPU. Consider using the `bgd` option which will'
                           ' run everything on the GPU.'))

        super().__init__(
            device=device,
            n_components=n_components,
            kernel=kernel,
            nu=nu,
        )

    def fit(  # type: ignore[override]
        self,
        x_ref: torch.Tensor,
        tol: float = 1e-6,
        max_iter: int = 1000,
        verbose: int = 0,
    ) -> Dict:
        """Fit the Nystroem approximation and Sklearn `SGDOneClassSVM` SVM model.

        Parameters
        ----------
        x_ref
            Training data.
        tol
            The decrease in loss required over the previous ``n_iter_no_change`` iterations in order to
            continue optimizing.
        max_iter
            The maximum number of optimization steps.
        verbose
            Verbosity level during training. ``0`` is silent, ``1`` a progress bar.

        Returns
        -------
        Dictionary with fit results. The dictionary contains the following keys:
            - converged: `bool` indicating whether training converged.
            - n_iter: number of iterations performed.
        """
        x_nys = self.nystroem.fit(x_ref).transform(x_ref)
        self.svm = SGDOneClassSVM(
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            nu=self.nu
        )
        x_nys = x_nys.cpu().numpy()
        self.svm = self.svm.fit(x_nys)
        self._set_fitted()
        return {
            'converged': self.svm.n_iter_ < max_iter,
            'n_iter': self.svm.n_iter_,
        }

    def format_fit_kwargs(self, fit_kwargs: Dict) -> Dict:
        """Format kwargs for `fit` method.

        Parameters
        ----------
        fit_kwargs
            dictionary of Kwargs to format. See `fit` method for details.

        Returns
        -------
        Formatted kwargs.
        """
        return dict(
            tol=fit_kwargs.get('tol', 1e-3),
            max_iter=fit_kwargs.get('max_iter', 1000),
            verbose=fit_kwargs.get('verbose', 0),
        )

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the score of `x`

        Parameters
        ----------
        x
            `torch.Tensor` with leading batch dimension.

        Returns
        -------
        `torch.Tensor¬` of scores with leading batch dimension.

        Raises
        ------
        NotFittedError
            Raised if method called and detector has not been fit.
        """
        self.check_fitted()
        x_nys = self.nystroem.transform(x)
        x_nys = x_nys.cpu().numpy()
        coef_ = self.svm.coef_ / (self.svm.coef_ ** 2).sum()
        x_nys = self.svm._validate_data(x_nys, accept_sparse="csr", reset=False)
        result = safe_sparse_dot(x_nys, coef_.T, dense_output=True).ravel()
        return - self._to_backend_dtype(result)


class BgdSVMTorch(SVMTorch):
    ensemble = False

    def __init__(
        self,
        nu: float,
        kernel: 'torch.nn.Module' = None,
        n_components: Optional[int] = None,
        device: TorchDeviceType = None,
    ):
        """Pytorch backend for the Support Vector Machine (SVM) outlier detector.

        Parameters
        ----------
        nu
            The proportion of the training data that should be considered outliers. Note that this does
            not necessarily correspond to the false positive rate on test data, which is still defined when
            calling the `infer_threshold` method.
        kernel
            Kernel function to use for outlier detection.
        n_components
            Number of components in the Nystroem approximation, by default uses all of them.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.
        """

        if (isinstance(device, str) and device == 'cpu') or \
                (isinstance(device, torch.device) and device.type == 'cpu'):
            warnings.warn(('The `bgd` optimization option is best suited for GPU. If '
                           'you want to use CPU, consider using the `sgd` option.'))

        super().__init__(
            device=device,
            n_components=n_components,
            kernel=kernel,
            nu=nu,
        )

    def fit(  # type: ignore[override]
        self,
        x_ref: torch.Tensor,
        step_size_range: Tuple[float, float] = (1e-8, 1.0),
        n_step_sizes: int = 16,
        tol: float = 1e-6,
        n_iter_no_change: int = 25,
        max_iter: int = 1000,
        verbose: int = 0,
    ) -> Dict:
        """Fit the Nystroem approximation and python SVM model.

        Parameters
        ----------
        x_ref
            Training data.
        step_size_range
            The range of values to be considered for the gradient descent step size at each iteration. This is
            specified as a tuple of the form `(min_eta, max_eta)`.
        n_step_sizes
            The number of step sizes in the defined range to be tested for loss reduction. This many points are spaced
            equidistantly along the range in log space.
        tol
            The decrease in loss required over the previous n_iter_no_change iterations in order to continue optimizing.
        n_iter_no_change
            The number of iterations over which the loss must decrease by `tol` in order for optimization to continue.
        max_iter
            The maximum number of optimization steps.
        verbose
            Verbosity level during training. ``0`` is silent, ``1`` a progress bar.

        Returns
        -------
        Dictionary with fit results. The dictionary contains the following keys:
            - converged: `bool` indicating whether training converged.
            - n_iter: number of iterations performed.
            - lower_bound: loss lower bound.
        """

        x_nys = self.nystroem.fit(x_ref).transform(x_ref)
        n, d = x_nys.shape
        min_eta, max_eta = step_size_range
        etas = torch.tensor(
            np.linspace(
                np.log(min_eta),
                np.log(max_eta),
                n_step_sizes
            ),
            dtype=x_nys.dtype,
            device=self.device
        ).exp()

        # Initialise coeffs/preds/loss
        coeffs = torch.zeros(d, dtype=x_nys.dtype, device=self.device)
        intercept = torch.zeros(1, dtype=x_nys.dtype, device=self.device)
        preds = x_nys @ coeffs + intercept
        loss = self.nu * (coeffs.square().sum()/2 + intercept) + hinge_loss(preds)
        min_loss, min_loss_coeffs, min_loss_intercept = loss, coeffs, intercept
        iter, t_since_improv = 0, 0
        converged = False

        with tqdm(total=max_iter, disable=not verbose) as pbar:
            while not converged:
                pbar.update(1)
                # First two lines give form of sgd update (for each candidate step size)
                sup_vec_inds = (preds < 1)
                cand_coeffs = coeffs[:, None] * \
                    (1-etas*self.nu) + etas*(x_nys[sup_vec_inds].sum(0)/n)[:, None]
                cand_intercept = intercept - etas*self.nu + (sup_vec_inds.sum()/n)

                # Compute loss for each candidate step size and choose the best
                cand_preds = x_nys @ cand_coeffs + cand_intercept
                cand_losses = self.nu * (cand_coeffs.square().sum(0)/2 + cand_intercept) + hinge_loss(cand_preds)
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
            'lower_bound': self._to_frontend_dtype(min_loss),
            'n_iter': iter
        }

    def format_fit_kwargs(self, fit_kwargs: Dict) -> Dict:
        """Format kwargs for `fit` method.

        Parameters
        ----------
        fit_kwargs
            dictionary of Kwargs to format. See `fit` method for details.

        Returns
        -------
        Formatted kwargs.
        """
        return dict(
            step_size_range=fit_kwargs.get('step_size_range', (1e-8, 1.0)),
            n_iter_no_change=fit_kwargs.get('n_iter_no_change', 25),
            tol=fit_kwargs.get('tol', 1e-6),
            verbose=fit_kwargs.get('verbose', 0),
            n_step_sizes=fit_kwargs.get('n_step_sizes', 16),
            max_iter=fit_kwargs.get('max_iter', 1000)
        )

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
        coeffs = torch.nn.functional.normalize(self.coeffs, dim=-1)
        preds = x_nys @ coeffs
        return -preds


class _Nystroem:
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
        n_components
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
