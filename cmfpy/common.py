"""
Common numeric routines.
"""
import numpy as np
import numpy.linalg as la
import numba

EPSILON = np.finfo(np.float).eps
FACTOR_MIN = 0


# @numba.jit(nopython=True, cache=True)
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


# @numba.jit(nopython=True, cache=True)
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


# @numba.jit(nopython=True, cache=True)
def tconv(W, H):
    """
    Tensor convolution.
    """
    I, J = W.shape[1], H.shape[1]
    Z = np.zeros((I, J))
    for lag, Wl in enumerate(W):
        Z += sdot(Wl, H, lag)
    return Z


# @numba.jit(nopython=True, cache=True)
def t_tconv(W, X):
    """
    Transposed tensor convolution.
    """
    I, J = W.shape[2], X.shape[1]
    Z = np.zeros((I, J))
    for lag, Wl in enumerate(W):
        Z += sdot(Wl.T, X, -lag)
    return Z
