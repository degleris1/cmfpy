"""
A Python implementation of seqNMF.

Written by Alex Williams and Anthony Degleris.
"""
import numpy as np
import matplotlib.pyplot as pyplot


# TODO: subclass np.ndarray?
class ShiftMatrix(object):
    """
    Thin wrapper around a numpy matrix to support shifting along the second
    axis and padding with zeros.
    """

    def __init__(self, X, L):
        """
        X : numpy 2-d array
        L : int, largest shift
        """

        # ShiftMatrix behaves like the original matrix
        self.shape = X.shape
        self.size = X.size

        # Padded version of X
        self.L = L
        self.X = np.pad(X, ((0, 0), (L, L)), mode='constant')

    def shift(self, l):
        if np.abs(l) > self.L:
            raise ValueError('requested too large of a shift.')

        r = slice(self.L - l, self.L + self.shape[1] - l)
        return self.X[:, r]

    def assign(self, Xnew):
        self.X[:, self.L:-self.L] = Xnew




    class ConvNMF(object):

    def __init__(self, n_components, maxlag, tol=1e-5, n_iter_max=100,
                 l1_W=0.0, l1_H=0.0):
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
        self.l1_W = l1_W
        self.l1_H = l1_H
        self._shifts = np.arange(maxlag*2 + 1) - maxlag
        self.loss_hist = None


    def fit(self, data, alg='mult'):
        # Check input
        if (data < 0).any():
            raise ValueError('Negative values in data to fit')

        data = ShiftMatrix(data, self.maxlag)
        m, n = data.shape

        # initialize W and H
        self.W = np.random.rand(self.maxlag*2 + 1, m, self.n_components)
        self.H = ShiftMatrix(np.random.rand(self.n_components, n), self.maxlag)


        # optimize
        if (alg == 'bcd'):
            self._fit_bcd(data, m, n, step_type='backtrack')
        elif (alg == 'mult'):
            self._fit_mult(data, m, n)

        return self



    def _fit_bcd(self, data, m, n, step_type='backtrack')

        converged, itr = False, 0

        # initial calculation of W gradient
        loss_1, grad_W = self._compute_gW(data)
        self.loss_hist = [loss_1]

        for itr in trange(self.n_iter_max):
            # update W
            step_W = self._scale_gW(grad_W, step_type)
            self.W = np.maximum(self.W - np.multiply(step_W, grad_W), 0)

            # compute gradient of H
            _, grad_H = self._compute_gH(data)

            # update W
            step_H = self._scale_gH(gradH, step_type)
            self.H.assign(np.maximum(self.H.shift(0) - np.multiply(step_H, grad_H), 0))

            # compute gradient of W
            loss_2, grad_W = self._compute_gW(data)
            self.loss_hist += [loss_2]

            # check convergence
            if (np.abs(loss_1 - loss_2) < self.tol):
                converged = True
                break
            else:
                loss_1 = loss_2


    def predict(self):
        """
        Return low-rank reconstruction of data.
        """

        # check that W and H are fit
        self._check_is_fitted()
        W, H = self.W, self. H

        # dimensions
        m, n = W.shape[1], H.shape[1]

        # preallocate result
        result = np.zeros((m, n))

        # iterate over lags
        for w, t in zip(W, self._shifts):
            result += np.dot(w, H.shift(t))

        return result


    def _check_is_fitted(self):
        if self.W is None or self.H is None:
            raise ValueError('This ConvNMF instance is not fitted yet.'
                             'Call \'fit\' with appropriate arguments '
                             'before using this method.')


    def _compute_loss(self, data):
        """
        Root Mean Squared Error
        """
        resid = (self.predict() - data.shift(0)).ravel()
        return np.sqrt(np.mean(np.dot(resid, resid)))


    def _compute_gW(self, data):

        # compute residuals
        resid = self.predict() - data.shift(0)

        # TODO: replace with broadcasting
        Wgrad = np.empty(self.W.shape)
        for l, t in enumerate(self._shifts):
            Wgrad[l] = np.dot(resid, self.H.shift(t).T)

        # compute loss
        r = resid.ravel()
        loss = np.sqrt(np.mean(np.dot(r, r)))

        return loss, Wgrad


    def _compute_gH(self, data):

        # compute residuals
        resid = self.predict() - data.shift(0)

        # compute gradient
        # TODO: speed up with broadcasting
        # Note: can sum up all the Wl.T first and then dot product
        Hgrad = np.zeros(self.H.shape)
        for l, t in enumerate(self._shifts):
            dh = np.dot(self.W[l].T, resid)
            Hgrad += _shift(dh, -t)

        # compute loss
        r = resid.ravel()
        loss = np.sqrt(np.mean(np.dot(r, r)))

        return loss, Hgrad


    def _scale_gW(self, grad_W, step_type):
        return 0.00001


    def _scale_gH(self, grad_H, step_type):
        return 0.00001


    # TODO: compute the lipschitz constant for optimal learning rate
    # TODO: backtracking line search


def _shift(X, l):
    """
    Shifts matrix X along second axis and zero pads
    """
    if l < 0:
        return np.pad(X, ((0, 0), (0, -l)), mode='constant')[:, -l:]
    elif l > 0:
        return np.pad(X, ((0, 0), (l, 0)), mode='constant')[:, :-l]
    else:
        return X
