"""
A Python implementation of CNMF.

Written by Alex Williams and Anthony Degleris.
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from tqdm import trange

from cnmfpy.conv import ShiftMatrix, tensor_conv, tensor_transconv
from cnmfpy.optimize import compute_gH, compute_gW
from cnmfpy.regularize import compute_smooth_kernel
from cnmfpy.algs import fit_bcd, fit_mult


class CNMF(object):

    def __init__(self, n_components, maxlag, tol=1e-5, n_iter_max=100,
                 l2_scfo=1e-6, l1_W=0.0, l1_H=0.0):
        """
        l1_W (float) : strength of sparsity penalty on W
        l1_H (float) : strength of sparsity penalty on H
        maxlag (int) : (L-1)/2
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
        
        self._shifts = np.arange(maxlag*2 + 1) - maxlag
        self._kernel = compute_smooth_kernel(maxlag)
        self.loss_hist = None


    def fit(self, data, alg='mult'):
        # Check input
        if (data < 0).any():
            raise ValueError('Negative values in data to fit')

        mag = np.amax(data)
        data = ShiftMatrix(data, self.maxlag)
        m, n = data.shape

        # initialize W and H
        self.W = mag * np.abs(np.random.rand(self.maxlag*2 + 1, m, self.n_components))
        self.H = ShiftMatrix(mag * np.abs(np.random.rand(self.n_components, n)), self.maxlag)

        # optimize
        if (alg == 'bcd'):
            fit_bcd(data, self)
        elif (alg == 'mult'):
            fit_mult(data, self)
        else:
            raise ValueError('No such algorithm found.')

        return self


    def predict(self):
        """
        Return low-rank reconstruction of data.
        """
        # check that W and H are fit
        self._check_is_fitted()

        return tensor_conv(self.W, self.H, self._shifts)
       

    def _check_is_fitted(self):
        if self.W is None or self.H is None:
            raise ValueError('This ConvNMF instance is not fitted yet.'
                             'Call \'fit\' with appropriate arguments '
                             'before using this method.')

