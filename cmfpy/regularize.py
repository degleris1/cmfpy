import numpy as np

from numpy.linalg import norm, multi_dot
from scipy.signal import convolve2d

from .conv import shift_cols, tensor_transconv


def compute_scfo_reg(data, W, H, kernel):
    # TODO where to apply operations???
    # penalize W
    pen_W = tensor_transconv(W, data)

    # smooth and penalize H
    smooth_H = _smooth(H.T, kernel)
    penalty = np.dot(pen_W, smooth_H)

    return norm(penalty)


def compute_scfo_gW(data, W, H, kernel):
    # TODO switch dot to multidot
    K, T = H.shape

    # preallocate
    sfco_gradW = np.empty(W.shape)

    # smooth H
    smooth_H_T = _smooth(H.T, kernel)

    not_eye = 1 - np.eye(K)

    # TODO: broadcast
    for l in np.arange(W.shape[0]):
        sfco_gradW[l] = multi_dot([shift_cols(data, -l),
                                   smooth_H_T[:T-l, :],
                                   not_eye])

    return sfco_gradW


def compute_scfo_gH(data, W, H, kernel):
    K, T = H.shape

    # W penalty
    pen_W = tensor_transconv(W, data)

    not_eye = 1 - np.eye(K)
    return np.dot(not_eye, _smooth(pen_W, kernel))


def compute_smooth_kernel(maxlag):
    # TODO check
    return np.ones([1, 2*maxlag+1])


def _smooth(X, kernel):
    return convolve2d(X, kernel, 'same')
