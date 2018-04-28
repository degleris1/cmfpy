import numpy as np


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



def tensor_conv(W, H, shifts):
    """
    Convolves a tensor W and ShiftMatrix H.
    """
   
    # preallocate result
    m, n = W.shape[1], H.shape[1]
    result = np.zeros((m, n))

    # TODO: replace with broadcasting
    # iterate over lags
    for w, t in zip(W, shifts):
        result += np.dot(w, H.shift(t))

    return result


def tensor_transconv(W, X, shifts):
    """
    Transpose tensor convolution of tensor W and ShiftMatrix X.
    """

    # preallocate result
    m, n = W.shape[2], X.shape[1]
    result = np.zeros((m, n))

    # TODO: replace with broadcasting
    # iterate over lags
    for w, t in zip(W, shifts):
        result += np.dot(w.T, X.shift(-t))

    return result



def shift_cols(X, l):
    """
    Shifts matrix X along second axis and zero pads
    """
    if l < 0:
        return np.pad(X, ((0, 0), (0, -l)), mode='constant')[:, -l:]
    elif l > 0:
        return np.pad(X, ((0, 0), (l, 0)), mode='constant')[:, :-l]
    else:
        return X