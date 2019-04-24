"""
A Python implementation of CNMF.
"""
import numpy as np
import numpy.random as npr
import numpy.linalg as la

import scipy.sparse

import time
from tqdm import trange

from . import algs
from . import initialize
from . import visualize

from .common import tconv

NOT_FITTED_ERROR = ValueError(
    "This CMF instance is not fitted yet. Call 'fit' with appropriate"
    "arguments before using this method."
)


class ModelDimensions:
    """Holds dimensions of CMF model."""

    def __init__(self, data=None, n_units=None, n_timepoints=None,
                 maxlag=None, n_components=None):
        """
        Parameters
        ----------
            data : ndarray
                Matrix holding time series, shape: (n_units, n_timepoints).
                If not specified, must provide both `n_units` and
                `n_timepoints`.
            n_units : int
                Number of features/measurements recorded in time series.
                Must be specified if `data` is not given.
            n_timepoints : int
                Length of time series. Must be specified if `data` is not
                given.
            maxlag : int
                Number of time steps in each temporal motifs/sequence. Must be
                specified.
            n_components : int
                Number of temporal motifs/sequences. Must be specified.
        """

        if data is None:
            if (n_units is None) or (n_timepoints is None):
                raise ValueError("Must either specify 'data' or ('n_units' "
                                 "and 'n_timepoints').")
        else:
            n_units, n_timepoints = data.shape

        if maxlag is None:
            raise ValueError("Must specify 'n_lags'.")

        if n_components is None:
            raise ValueError("Must specify 'n_components'.")

        self.n_units = n_units
        self.n_timepoints = n_timepoints
        self.maxlag = maxlag
        self.n_components = n_components

    def __iter__(self):
        """Iterator over model dimensions."""
        yield from self.__dict__.items()


class CMF(object):
    """
    Convolutive Matrix Factorization (CNMF)

    Factors a matrix into a convolution between a tensor `W` and a
    matrix `H`.
    """

    def __init__(self, n_components, maxlag, l1=(0.0, 0.0), l2=(0.0, 0.0),
                 method='mult', init='rand', sort_components=True,
                 rand_state=None, **alg_opts):
        """
        Parameters
        ----------
        n_components : int
            Number of components to fit.
        maxlag : int
            Maximum time lag in each sequence. A single sequence can lag up to
            `maxlag` entries right of the first time point.
        tol : float, optional
            Tolerance for convergence. If the change in cost is less than the
            `tol`, the algorithm will terminate early.
        l1 : tuple of floats, optional
            Weights of the L1 regularization for `W` and `H`, respectively.
            Default: (0.0, 0.0)
        l2 : tuple of floats, optional
            Weights of the L2 regularization for `W` and `H`, respectively.
            Default: (0.0, 0.0)
        method : str
            String specifying fitting algorithm.
        init : str
            String specifying initialization method.
        sort_components : bool
            If True, components are sorted by size after optimization
        rand_state : None, int, or RandomState
            Seed for random number generator.
        **alg_opts : dict
            Additional keyword arguments are passed to the fitting algorithm.
                - max_iter (int) : max number of iterations.
                - tol (float) : convergence criterion.
                - verbose (bool) : If True, log output to stdout.
        """
        self.n_components = n_components
        self.maxlag = maxlag

        self.l1 = l1
        self.l2 = l2

        self.init = init
        self.method = method

        self.sort_components = sort_components

        self.alg_opts = alg_opts

        self.alg_opts["l1_W"] = l1[0]
        self.alg_opts["l1_H"] = l1[1]
        self.alg_opts["l2_W"] = l2[0]
        self.alg_opts["l2_H"] = l2[1]

        if isinstance(rand_state, np.random.RandomState):
            self._rs = rand_state
        else:
            self._rs = np.random.RandomState(rand_state)

    def fit(self, data):
        """
        Fits model to the data.

        Parameters
        ----------
        data : array-like, shape (n_neurons, n_time)
            Training data to fit.

        Returns
        -------
        self : object
            Returns the model instance.
        """

        # Check input
        if (data < 0).any():
            raise ValueError('Negative values in data to fit')

        # Pad data.
        data = _safe_zero_pad(data, self.maxlag)

        # Determine dimensions of model.
        dims = ModelDimensions(
            data, maxlag=self.maxlag, n_components=self.n_components)

        # Initialize W and H
        W0, H0 = getattr(initialize, self.init)(data, dims, self._rs)

        # Get fitting algorithm.
        self.W_, self.H_, raw_info = \
            getattr(algs, self.method)(data, W0, H0, **self.alg_opts)
        self.info_ = dict(raw_info)

        # Undo zero-padding
        self.H_ = self.H_[:, self.maxlag:-self.maxlag]

        # Sort components by size.
        if self.sort_components:
            idx = np.argsort(
                la.norm(self.W_, axis=(0, 1)) * la.norm(self.H_, axis=1))
            self.W_ = self.W_[:, :, idx]
            self.H_ = self.H_[idx, :]

    def _assert_fitted(self):
        """Checks that W_ and H_ exist."""
        if not hasattr(self, "W_"):
            raise NOT_FITTED_ERROR

    def argsort_units(self):
        """Returns ordering of units/features based on soft clustering."""

        # For each unit, find the component with the largest weight.
        mx_k = np.argmax(np.sum(self.W_, axis=0), axis=-1)

        # Find the lag of each unit's peak activity on the max component.
        pk = np.argmax(self.W_[:, np.arange(self.n_units), mx_k], axis=0)

        # Lexographically sort units by component and time lag.
        return np.lexsort((pk, mx_k))

    def predict(self):
        """
        Computes low-rank reconstruction of data.

        Returns
        -------
        est : array-like, shape (n_time, n_units)
            Model estimate of the data.
        """
        self._assert_fitted()
        return tconv(self.W_, self.H_)

    def score(self, data):
        """
        Return the R^2 score, defined as the one minus the squared norm of the
        error divided by the square norm of the data.

        Parameters
        ----------
        data : array-like, shape (n_time, n_units)
            Data to fit when scoring.

        Returns
        -------
        score : float
            A score for how well the model fits the data. A score of 1 is
            perfect.

        TODO: demean the data
        """
        error = self.predict() - data
        return 1 - (la.norm(error)**2 / la.norm(data)**2)

    @property
    def n_units(self):
        self._assert_fitted()
        return self.W_.shape[1]

    @property
    def n_timepoints(self):
        self._assert_fitted()
        return self.H_.shape[1]

    def plot(self, data=None, sort=True, **kwargs):
        """
        Visualizes data and model components.
        """
        self._assert_fitted()
        W, H = self.W_, self.H_

        # Plot model prediction is data is not provided
        data = self.predict() if data is None else data

        # Reorder units by model factors
        idx = self.argsort_units() if sort else np.arange(self.n_units)

        # Call plotting function.
        return visualize.plot_result(data[idx], W[:, idx, :], H, **kwargs)


def _safe_zero_pad(X, L):
    """
    Zero pad matrix X with L columns of zeros, preserving
    sparse matrix formats.
    """
    N = X.shape[0]

    if scipy.sparse.isspmatrix(X):
        pd = scipy.sparse.csr_matrix((N, L))
        return scipy.sparse.vstack((pd, X, pd))

    elif isinstance(X, np.ndarray):
        return np.pad(X, ((0, 0), (L, L)), mode="constant")
