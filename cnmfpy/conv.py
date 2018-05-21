import numpy as np

# TODO: subclass np.ndarray?
class ShiftMatrix(object):
    def __init__(self, X, L):
        """
        Shift Matrix.
        
        Thin wrapper around a numpy ndarray to support shifting along the second
        axis and padding with zeros.

        Parameters
        ----------
        X : numpy, shape (n1, n2)
            Numpy array to be wrapped.
        L : int
            Largest shift in either direction.
        """
        # behaves like the original ndarray
        self.shape = X.shape
        self.size = X.size

        # Padded version of X
        self.L = L
        self.X = np.pad(X, ((0, 0), (L, L)), mode='constant')


    def shift(self, l):
        """
        Shifts the columns left, padding with zeros on the right.

        Parameters
        ----------
        l : int
            Number of times to shift left.

        Returns
        -------
        X_shifted : ndarray
            Returns a shifted version of the original ndarray.
        """
        if np.abs(l) > self.L:
            raise ValueError('requested too large of a shift.' + str(np.abs(l)) + ' > ' + str(self.L))

        r = slice(self.L - l, self.L + self.shape[1] - l)
        return self.X[:, r]


    def assign(self, Xnew):
        """
        Reassigns the numpy ndarray.

        Parameters
        ----------
        Xnew : ndarray, shape (n1, n2)
            A new ndarray of the same shape as before.
        """
        self.X[:, self.L:-self.L] = Xnew



def tensor_conv(W, H):
    """
    Convolves a tensor W and ShiftMatrix H.

    Parameters
    ----------
    W : ndarray, shape (n_lag, n_neurons, n_components)
        Tensor of neural sequences.
    H : ShiftMatrix, shape (n_components, n_time)
        ShiftMatrix of time componenents.

    Returns
    -------
    X : ndarray, shape (n_neurons, n_time)
        Tensor convolution of W and H.
    """   
    # preallocate result
    m, n = W.shape[1], H.shape[1]
    result = np.zeros((m, n))

    # TODO: replace with broadcasting
    # iterate over lags
    shifts = np.arange(W.shape[0])
    for w, t in zip(W, shifts):
        result += np.dot(w, H.shift(t))

    return result


def tensor_transconv(W, X):
    """
    Transpose tensor convolution of tensor W and ShiftMatrix X.

    Parameters
    ----------
    W : ndarray, shape (n_lag, n_neurons, n_components)
        Tensor of neural sequences.
    H : ShiftMatrix, shape (n_components, n_time)
        ShiftMatrix of time componenents.

    Returns
    -------
    X : ndarray, shape (n_neurons, n_time)
        Transpose tensor convolution of W and H, i.e. the tensor convolution
        but shifted the opposite direction.
    """
    # preallocate result
    m, n = W.shape[2], X.shape[1]
    result = np.zeros((m, n))

    # TODO: replace with broadcasting
    # iterate over lags
    shifts = np.arange(W.shape[0])
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