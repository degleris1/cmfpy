"""
Common numeric routines.
"""
import numpy as np
import numpy.linalg as la
import numba

EPSILON = np.finfo(np.float).eps
FACTOR_MIN = 0


@numba.jit(nopython=True, cache=True)
def sdot_plus(A, B, shift, Z):
    """
    Returns matrix multiplication between A and B, shifting columns of B.

    shift < 0, shifts columns to left.
    shift > 0, shifts columns to right.
    """
    if shift < 0:
        Z[:, :shift] += np.dot(A, B[:, (-shift):])
    elif shift > 0:
        Z[:, shift:] += np.dot(A, B[:, :(-shift)])
    else:
        Z += np.dot(A, B)   # shift == 0
    return Z


@numba.jit(nopython=True, cache=True)
def sdot(A, B, shift):
    """
    Returns matrix multiplication between A and B, shifting columns of B.

    shift < 0, shifts columns to left.
    shift > 0, shifts columns to right.
    """
    if shift < 0:
        AB = np.dot(A, B[:, (-shift):])
        Z = np.zeros((A.shape[0], -shift))
        return np.column_stack((AB, Z))
    elif shift > 0:
        AB = np.dot(A, B[:, :(-shift)])
        Z = np.zeros((A.shape[0], shift))
        return np.column_stack((Z, AB))
    else:
        return np.dot(A, B)   # shift == 0


@numba.jit(nopython=True, cache=True)
def t_sdot(A, B, shift):
    """
    Returns matrix multiplication between A and transpose(B), shifting
    columns of B.

    shift < 0, shifts columns of B to left. Rows of B.T up.
    shift > 0, shifts columns of B to right. Rows of B.T down.
    """
    if shift < 0:
        return np.dot(A[:, :shift], B.T[(-shift):])

    elif shift > 0:
        return np.dot(A[:, shift:], B.T[:(-shift)])

    else:
        return np.dot(A, B.T)   # shift == 0


@numba.jit(nopython=True, cache=True)
def tconv3(W, H, Z):
    """Tensor convolution"""
    L, N, K = W.shape
    Z.fill(0.0)
    for lag in range(L):
        Z += sdot(W[lag], H, lag)
    return Z


@numba.jit(nopython=True, cache=True)
def tconv(W, H):
    """Tensor convolution"""
    Z = np.empty((W.shape[1], H.shape[1]))
    return tconv3(W, H, Z)


@numba.jit(nopython=True, cache=True)
def t_tconv3(W, X, Z):
    """Transposed tensor convolution"""
    L, N, K = W.shape
    Z.fill(0.0)
    for lag in range(L):
        Z += sdot(W[lag].T, X, -lag)
    return Z


@numba.jit(nopython=True, cache=True)
def t_tconv(W, H):
    """Transposed tensor convolution"""
    Z = np.empty((W.shape[2], H.shape[1]))
    return t_tconv3(W, H, Z)
