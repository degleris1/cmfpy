import numpy as np

from ..common import cmf_predict, tensor_transconv, shift_cols
from .base import AbstractOptimizer

# TODO float or float32?
EPSILON = np.finfo(np.float).eps


class MultUpdate(AbstractOptimizer):
    """
    Multiplicative update rule.
    """
    def __init__(self, data, dims, patience=3, tol=1e-5, **kwargs):
        super().__init__(data, dims, patience=patience, tol=tol, **kwargs)
        self.stochastic = kwargs.get('stochastic', False)
        if self.stochastic:
            if not 'window_size' in kwargs:
                raise AttributeError("Must pass keyword argument window_size to use stochastic updates")
            self.window_size = kwargs['window_size']

    def update(self):
        if self.stochastic:
            X = self.X
            H = self.H
            W = self.W
            window_size = self.window_size

            (_, T) = X.shape
            start_idx = np.random.randint(T - window_size + 1)
            X = X[:, start_idx:start_idx + window_size]
            H = H[:, start_idx:start_idx + window_size]

            num_W, denom_W = self._compute_mult_W(X, W, H)

            # The stochastic W update is unchanged – we still update every entry
            self.W = np.divide(np.multiply(self.W, num_W), denom_W + EPSILON)

            num_H, denom_H = self._compute_mult_H(X, W, H)

            # For the stochastic H update we only update a subset of entries
            H_new = np.divide(np.multiply(H, num_H), denom_H + EPSILON)
            self.H[:, start_idx:start_idx + window_size] = H_new

        else:
            # update W
            num_W, denom_W = self._compute_mult_W()
            self.W = np.divide(np.multiply(self.W, num_W), denom_W + EPSILON)

            # update H
            num_H, denom_H = self._compute_mult_H()
            self.H = np.divide(np.multiply(self.H, num_H), denom_H + EPSILON)

        self.cache_resids()
        return self.loss

    def _compute_mult_W(self, X = None, W = None, H = None):
        X = X if X is not None else self.X
        W = W if W is not None else self.W
        H = H if H is not None else self.H

        # preallocate
        num = np.zeros(W.shape)
        denom = np.zeros(W.shape)

        est = cmf_predict(W, H)

        # TODO: broadcast
        for l in np.arange(W.shape[0]):
            # TODO switch to sdot
            num[l] = np.dot(X[:, l:], shift_cols(H, l).T)
            denom[l] = np.dot(est[:, l:], shift_cols(H, l).T)

        return num, denom

    def _compute_mult_H(self, X = None, W = None, H = None):
        X = X if X is not None else self.X
        W = W if W is not None else self.W
        H = H if H is not None else self.H

        est = cmf_predict(W, H)

        num = tensor_transconv(W, X)
        denom = tensor_transconv(W, est)

        return num, denom
