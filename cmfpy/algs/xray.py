"""
Extension of XRAY algorithm for convolutive NMF model.
"""
import numba
import numpy as np
import numpy.linalg as la

from ..common import cmf_predict
from .base import AbstractOptimizer
from .hals import _setup_batches, _setup_H_update, _update_H


class XRAY(AbstractOptimizer):
    """
    Class implementing XRAY algorithm for convolutive NMF.

    Reference
    ---------
    Kumar A, Sindhwani V, Kambadur P (2013). Fast Conical Hull Algorithms for
        Near-separable Non-negative Matrix Factorization. ICML.
    """

    def __init__(self, data, dims, patience=3, tol=1e-5, **kwargs):
        super().__init__(data, dims, patience=patience, tol=tol, **kwargs)

        self.batch_sizes, self.batch_inds = _setup_batches(self.n_components,
                                                           self.n_timepoints,
                                                           self.maxlag)

    def init_params(self):
        """Override initialization, self.update(...) takes care of it."""
        pass

    def update(self):

        # Initialize storage for W and H
        self.W = np.zeros((self.maxlag, self.n_features, self.n_components))
        self.H = np.empty((self.n_components, self.n_timepoints))

        # Initialize residual matrix
        residual = self.X.copy()

        # Iteratively solve each component in the model.
        for k in range(self.n_components):

            # Assign next W component to extreme ray.
            colnorms = np.linalg.norm(residual, axis=0) ** 2
            t = _conv_argmax(colnorms, self.maxlag)
            self.W[:, :, k] = self.X[:, t:(t + self.maxlag)].T

            # Fit best H for last W component
            self.W_norms, self.W_raveled, self.W_clones = _setup_H_update(
                self.W,
                self.batch_sizes
            )
            _update_H(self.W, self.H, self.resids,
                      self.W_norms, self.W_raveled, self.W_clones,
                      self.batch_inds, self.batch_sizes)

            # Update residual.
            residual -= cmf_predict(self.W[:, :, [k]], self.H[[k]])

            # TODO: update all W and H components <k ??

        # need to compute loss manually
        return la.norm(residual) / la.norm(self.X)

    def converged(self, loss_hist):
        """Algorithm converges after one iteration."""
        return self.W is not None


@numba.jit(nopython=True)
def _conv_argmax(x, maxlag):
    # Index interating over elements of x
    i = 0

    # Compute running sum over maxlag elements
    rs = 0.0
    while i < maxlag:
        rs += x[i]
        i += 1

    # Store largest running sum
    mx = rs   # stores max
    amx = 0   # stores argmax

    # Iterate over remaining elements in x
    # TODO what happens when i > len(x) - maxlag?
    while i < len(x):
        # Recompute moving sum within window.
        rs += x[i] - x[i - maxlag]

        # Save largest running sum.
        if rs > mx:
            mx = rs
            amx = i - maxlag

        # Move to next timepoint.
        i += 1

    return amx
