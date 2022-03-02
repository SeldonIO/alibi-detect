import logging
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union, Type
from alibi_detect.cd.base import BaseContextAwareDrift
from alibi_detect.cd.domain_clf import DomainClf, SVCDomainClf
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ContextAwareDriftTF(BaseContextAwareDrift):
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
            domain_clf: Type[DomainClf] = SVCDomainClf,
            n_permutations: int = 1000,
            cond_prop: float = 0.25,
            lams: Optional[Tuple[float, float]] = None,
            batch_size: Optional[int] = 256,
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
            being test instances). Must be a subclass of DomainClf. # TODO - add link
        n_permutations
            Number of permutations used in the permutation test.
        cond_prop
            Proportion of contexts held out to condition on.
        lams
            Ref and test regularisation parameters. Tuned if None.
        batch_size
            If not None, then compute batches of MMDs at a time (rather than all at once).
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
        self.meta.update({'backend': 'tensorflow'})
        raise NotImplementedError('ContextAwareDrift not yet implemented for tensorflow backend.')

    def score(self,  # type: ignore[override]
              x: Union[np.ndarray, list], c: np.ndarray) -> Tuple[float, float, np.ndarray, Tuple]:
        """
        Compute the p-value resulting from a permutation test using the maximum mean discrepancy
        as a distance measure between the reference data and the data to be tested.  # TODO

        Parameters
        ----------
        x
            Batch of instances.
        c
            Context associated with batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the MMD^2 between the reference and test set, the
        MMD^2 values from the permutation test, and the coupling matrices.
        """
        pass
