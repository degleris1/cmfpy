"""
A Python implementation of seqNMF.

Written by Alex Williams and Anthony Degleris.
"""
import numpy as np
import matplotlib.pyplot as plt

EPSILON = np.finfo(np.float32).eps


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


    def fit(self, data, alg='bcd'):
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
            self._fit_bcd(data, step_type='backtrack')
        elif (alg == 'mult'):
            self._fit_bcd(data, step_type='mult')

        return self



    def _fit_bcd(self, data, step_type='backtrack'):
        m, n = data.shape
        converged, itr = False, 0

        # initial calculation of W gradient
        loss_1, grad_W = self._compute_gW(data)
        self.loss_hist = [loss_1]

        for itr in range(self.n_iter_max):
            # update W
            step_W = self._scale_gW(grad_W, step_type)
            self.W = np.maximum(self.W - np.multiply(step_W, grad_W), 0)

            # compute gradient of H
            _, grad_H = self._compute_gH(data)

            # update W
            step_H = self._scale_gH(grad_H, step_type)
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

        return _tensor_conv(self.W, self.H, self._shifts)
       

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
        return np.sqrt(np.mean(np.multiply(resid, resid)))  # TODO: multiply or dot


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
        #Hgrad = np.zeros(self.H.shape)
        #for l, t in enumerate(self._shifts):
        #    dh = np.dot(self.W[l].T, resid)
        #    Hgrad += _shift(dh, -t)

        # compute grad
        resid = ShiftMatrix(resid, self.maxlag)
        Hgrad = _tensor_transconv(self.W, resid, self._shifts)
        
        # compute loss
        r = resid.shift(0).ravel()
        loss = np.sqrt(np.mean(np.dot(r, r)))

        return loss, Hgrad


    def _scale_gW(self, grad_W, step_type):
        if (step_type == 'backtrack'):
            step_W = 0.00001

        elif (step_type == 'mult'):
            # preallocate
            step_W = np.zeros(grad_W.shape)

            estimate = self.predict()
            H, W = self.H, self.W
            for l, t in enumerate(self._shifts):
                step_W[l] = np.divide(W[l], np.dot(estimate, H.shift(t).T) + EPSILON)
            
        else:
            raise ValueError('Invalid BCD step type.')

        return step_W


    def _scale_gH(self, grad_H, step_type):
        if (step_type == 'backtrack'):
            step_H = 0.00001

        elif (step_type == 'mult'):
            estimate = ShiftMatrix(self.predict(), self.maxlag)
            W, H, shifts = self.W, self.H.shift(0), self._shifts
            step_H = np.divide(H, _tensor_transconv(W, estimate, shifts) + EPSILON)

        else:
            raise ValueError('Invalid BCD step type.')

        return step_H

    # TODO: compute the lipschitz constant for optimal learning rate
    # TODO: backtracking line search



def _tensor_conv(W, H, shifts):
    """
    Convolves a tensor W and ShiftMatrix H.
    """
   
    # preallocate result
    m, n = W.shape[1], H.shape[1]
    result = np.zeros((m, n))

    # iterate over lags
    for w, t in zip(W, shifts):
        result += np.dot(w, H.shift(t))

    return result


def _tensor_transconv(W, X, shifts):
    """
    Transpose tensor convolution of tensor W and ShiftMatrix X.
    """

    # preallocate result
    m, n = W.shape[2], X.shape[1]
    result = np.zeros((m, n))

    # iterate over lags
    for w, t in zip(W, shifts):
        result += np.dot(w.T, X.shift(-t))

    return result



    



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


def seq_nmf_data(N, T, L, K):
    """Creates synthetic dataset for conv NMF
    Args
    ----
    N : number of neurons
    T : number of timepoints
    L : max sequence length
    K : number of factors / rank
    Returns
    -------
    data : N x T matrix
    """

    # low-rank data
    W, H = np.random.rand(N, K), np.random.rand(K, T)
    W[W < .5] = 0
    H[H < .8] = 0
    lrd = np.dot(W, H)

    # add a random shift to each row
    lags = np.random.randint(0, L, size=N)
    data = np.array([np.roll(row, l, axis=-1) for row, l in zip(lrd, lags)])
    # data = lrd

    return data, W, H


if (__name__ == '__main__'):
    data, W, H = seq_nmf_data(100, 300, 10, 2)

    losses = []

    for k in range(1, 5):
        model = ConvNMF(k, 15).fit(data, alg='mult')
        plt.plot(model.loss_hist)
        losses.append(model.loss_hist[-1])

    plt.figure()
    plt.plot(losses)
    plt.show()