"""
A Python implementation of CNMF.

Written by Alex Williams and Anthony Degleris.
"""
import numpy as np

from cnmfpy.conv import tensor_conv
from cnmfpy.optimize import compute_loadings
from cnmfpy.regularize import compute_smooth_kernel
from cnmfpy.algs import fit_bcd, fit_mult


class CNMF(object):
    def __init__(self, n_components, maxlag, tol=1e-5, n_iter_max=100,
                 l2_scfo=0, l1_W=0.0, l1_H=0.0, method='mult'):
        """
        Convolutive Non-Negative Matrix Factorization (CNMF)

        Factors a matrix into a convolution between a tensor `W` and a
        matrix `H`.

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
        n_iter_max : int, optional
            Maximum number of iterations during algorithm fitting.
        l2_scfo : float, optional
            Weight of the soft cross-factors orthogonality regularizer. See
            references for details.
        l1_W : float, optional
            Weight of the L1 regularizer for the entries of `W`.
        l1_H : float, optional
            Weight of the L1 regularizer for the entries of `H`.

        References
        ----------
        See Mackevicius, Bahle, et al., *Unsupervised discovery of temporal
        sequences in high-dimensional datasets, with applications to
        neuroscience.*
        """
        self.n_components = n_components
        self.maxlag = maxlag

        self.W = None
        self.H = None

        self.tol = 1e-4
        self.n_iter_max = n_iter_max

        self.l2_scfo = l2_scfo
        self.l1_W = l1_W
        self.l1_H = l1_H

        self._kernel = compute_smooth_kernel(maxlag)
        self.loss_hist = None
        self.method = method

    def fit(self, data):
        """
        Fit a CNMF model to the data.

        Parameters
        ----------
        data : array-like, shape (n_neurons, n_time)
            Training data to fit.
        alg : string {'mult', 'bcd'}, optional
            Algorithm used to fit the data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check input
        if (data < 0).any():
            raise ValueError('Negative values in data to fit')

        mag = np.amax(data)
        n_neurons, n_time = data.shape

        # initialize W and H
        self.W = mag * np.abs(np.random.rand(self.maxlag, n_neurons,
                                             self.n_components))
        self.H = mag * np.abs(np.random.rand(self.n_components, n_time))

        # optimize
        if (self.method == 'bcd_backtrack'):
            fit_bcd(data, self, step_type='backtrack')
        elif (self.method == 'bcd_const'):
            fit_bcd(data, self, step_type='constant')
        elif (self.method == 'mult'):
            fit_mult(data, self)
        else:
            raise ValueError('No such algorithm found.')

        # compute explanatory power of each factor
        loadings = compute_loadings(data, self.W, self.H)

        # sort factors by power
        ind = np.argsort(loadings)
        self.W = self.W[:, :, ind]
        self.H = self.H[ind, :]

        return self

    def predict(self):
        """
        Return low-rank reconstruction of data.

        Returns
        -------
        est : array-like, shape (n_time, n_features)
            Reconstruction of the data using `W` and `H`.
        """
        # check that W and H are fit
        self._check_is_fitted()

        return tensor_conv(self.W, self.H)

    def _check_is_fitted(self):
        """
        Check if `W`, `H` have been fitted.
        """
        if self.W is None or self.H is None:
            raise ValueError('This ConvNMF instance is not fitted yet.'
                             'Call \'fit\' with appropriate arguments '
                             'before using this method.')
