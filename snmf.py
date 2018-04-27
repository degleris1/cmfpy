"""
A Python implementation of seqNMF.

Written by Alex Williams and Anthony Degleris.
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from tqdm import trange

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
                 l1_W=0.0001, l1_H=0.0):
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

        mag = np.amax(data)
        data = ShiftMatrix(data, self.maxlag)
        m, n = data.shape

        # initialize W and H
        self.W = mag * np.abs(np.random.rand(self.maxlag*2 + 1, m, self.n_components))
        self.H = ShiftMatrix(mag * np.abs(np.random.rand(self.n_components, n)), self.maxlag)


        # optimize
        if (alg == 'bcd'):
            self._fit_bcd(data, step_type='backtrack')
        elif (alg == 'mult'):
            self._fit_mult(data)
        elif (alg == 'mult_bcd'):
            self._fit_bcd(data, step_type='mult')

        return self



    def _fit_bcd(self, data, step_type='backtrack'):
        m, n = data.shape
        converged, itr = False, 0

        # initial calculation of W gradient
        loss_1, grad_W = self._compute_gW(data)
        self.loss_hist = [loss_1]

        for itr in trange(self.n_iter_max):
            # update W
            step_W = self._scale_gW(data, grad_W, step_type)
            new_W = np.maximum(self.W - np.multiply(step_W, grad_W), 0)
            self.W = _soft_thresh(new_W, self.l1_W)

            # compute gradient of H
            _, grad_H = self._compute_gH(data)

            # update H
            step_H = self._scale_gH(data, grad_H, step_type)
            new_H = np.maximum(self.H.shift(0) - np.multiply(step_H, grad_H), 0)
            self.H.assign(_soft_thresh(new_H, self.l1_H))

            # compute gradient of W
            loss_2, grad_W = self._compute_gW(data)
            self.loss_hist += [loss_2]

            # check convergence
            if (np.abs(loss_1 - loss_2) < self.tol):
                converged = True
                break
            else:
                loss_1 = loss_2


    def _fit_mult(self, data):
        m, n = data.shape
        converged, itr = False, 0

        # initial loss
        self.loss_hist = [self._compute_loss(data)]

        for itr in trange(self.n_iter_max):
            # update W
            mult_W = self._compute_mult_W(data)
            self.W = np.multiply(self.W, mult_W)
            self.loss_hist += [self._compute_loss(data)]

            # update h
            mult_H = self._compute_mult_H(data)
            self.H.assign(np.multiply(self.H.shift(0), mult_H))
            self.loss_hist += [self._compute_loss(data)]

            # check convergence
            loss_1, loss_2 = self.loss_hist[-2:]
            if (np.abs(loss_1 - loss_2) < self.tol):
                converged = True
                break



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


    def _scale_gW(self, data, grad_W, step_type):
        if (step_type == 'backtrack'):
            step_W = self._backtrack(data, grad_W, 0)

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


    def _scale_gH(self, data, grad_H, step_type):
        if (step_type == 'backtrack'):
            step_H = self._backtrack(data, 0, grad_H)

        elif (step_type == 'mult'):
            estimate = ShiftMatrix(self.predict(), self.maxlag)
            W, H, shifts = self.W, self.H.shift(0), self._shifts
            step_H = np.divide(H, _tensor_transconv(W, estimate, shifts) + EPSILON)

        else:
            raise ValueError('Invalid BCD step type.')

        return step_H


    def _backtrack(self, data, grad_W, grad_H, beta=0.8, alpha=0.00001, max_iters=500):
        """
        Backtracking line search to find a step length.
        """
        W, H = self.W.copy(), self.H.shift(0).copy()
        loss = self._compute_loss(data)
        grad_mag = la.norm(grad_W)**2 + la.norm(grad_H)**2
        t = 1.0

        # first check
        self.H.assign(H - t*grad_H)
        self.W = W - t*grad_W

        iters = 1

        # backtrack
        while (self._compute_loss(data) > loss - alpha * t * grad_mag
               and iters < max_iters):
            t = beta * t
            self.H.assign(np.maximum(H - t*grad_H, 0))
            self.W = np.maximum(W - t*grad_W, 0)
            iters += 1

        #if (iters == max_iters):
        #    print('Backtracking did not converge.')

        # reset W, H
        self.H.assign(H)
        self.W = W

        return t



    def _compute_mult_W(self, data):
        # preallocate
        mult_W = np.zeros(self.W.shape)

        H = self.H
        estimate = self.predict()

        # TODO: broadcast
        for l, t in enumerate(self._shifts):
            num = np.dot(data.shift(0), H.shift(t).T)
            denom = np.dot(estimate, H.shift(t).T)
            mult_W[l] = np.divide(num, denom + EPSILON)

        return mult_W


    def _compute_mult_H(self, data):
        W = self.W
        estimate = ShiftMatrix(self.predict(), self.maxlag)

        num = _tensor_transconv(W, data, self._shifts)
        denom = _tensor_transconv(W, estimate, self._shifts)
       
        return np.divide(num, denom + EPSILON)


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


def _soft_thresh(X, l):
    """
    Soft thresholding function for sparsity.
    """
    return np.maximum(X-l, 0) - np.maximum(-X-l, 0)





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
    W[W < .8] = 0
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

    K = 3
    for k in range(1, K+1):
        model = ConvNMF(k, 10).fit(data, alg='bcd')
        plt.plot(model.loss_hist[1:])
        losses.append(model.loss_hist[-1])

    plt.figure()
    plt.plot(range(1,K+1), losses)
    plt.show()


    plt.figure()
    plt.imshow(model.predict())
    plt.title('Predicted')

    plt.figure()
    plt.imshow(data)

    plt.show()