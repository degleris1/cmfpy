import numpy as np
import numba


#@numba.jit
def tensor_conv(W, H):
    """
    Convolves a tensor W and matrix H.

    Parameters
    ----------
    W : ndarray, shape (n_lag, n_neurons, n_components)
        Tensor of neural sequences.
    H : ndarray, shape (n_components, n_time)
        Matrix of time componenents.

    Returns
    -------
    X : ndarray, shape (n_neurons, n_time)
        Tensor convolution of W and H.
    """
    L = W.shape[0]

    return np.dot(hunfold(W), shift_and_stack(H, L))


#@numba.jit
def tensor_transconv(W, X):
    """
    Transpose tensor convolution of tensor `W` and matrix `X`.

    Parameters
    ----------
    W : ndarray, shape (n_lag, n_neurons, n_components)
        Tensor of neural sequences.
    X : ndarray, shape (n_components, n_time)
        Matrix of time componenents.

    Returns
    -------
    X : ndarray, shape (n_neurons, n_time)
        Transpose tensor convolution of W and X, i.e. the tensor convolution
        but shifted the opposite direction.

    Notes
    -----
    # TODO: Consider speed up
    """
    L, N, K = W.shape
    T = X.shape[1]

    result = np.zeros((K, T))
    for lag, w in enumerate(W):
        result[:, :T-lag] += np.dot(w.T, shift_cols(X, -lag))

    return result


#@numba.jit(nopython=True, cache=True)
def shift_cols(X, lag):
    """
    Shifts the columns of a matrix `X` right by `l` lags. Drops columns that
    are shifted beyond the dimension of the matrix.

    # TODO: remove cases
    """
    if (lag <= 0):
        return X[:, -lag:]
    else:  # lag > 0
        return X[:, :-lag]


def pad_shift_cols(X, l):
    """
    Shifts matrix X along second axis and zero pads

    TODO: reduce cases
    """
    if l < 0:
        return np.pad(X, ((0, 0), (0, -l)), mode='constant')[:, -l:]
    elif l > 0:
        return np.pad(X, ((0, 0), (l, 0)), mode='constant')[:, :-l]
    else:
        return X


#@numba.jit
def hunfold(W):
    """
    Unfold a tensor along its first mode, stacking horizontally.
    """
    L, N, K = W.shape
    return np.swapaxes(W, 1, 2).reshape((K*L, N)).T


#@numba.jit
def hunfold_trans(M):
    """
    Unfold a tensor along its first mode, stacking the transpose of each
    slice horizontally.
    """
    L, N, K = M.shape
    return M.reshape((L*N, K)).T


#@numba.jit
def shift_and_stack(H, L):
    """
    Vertically stack several shifted copies of H.
    """
    K, T = H.shape

    H_stacked = np.zeros((L*K, T))
    for lag in range(L):
        H_stacked[K*lag:K*(lag+1), lag:] = shift_cols(H, lag)

    return H_stacked
