import numpy as np

from numpy.linalg import norm
from scipy.signal import convolve2d
from cnmfpy.conv import shift_cols, tensor_transconv


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
    smooth_H = _smooth(H.T, kernel)

    not_eye = np.ones((K, K)) - np.eye(K)

    # TODO: broadcast
    for l in np.arange(W.shape[0]):
        sfco_gradW[l] = shift_cols(data, -l).dot(smooth_H).dot(not_eye)

    return sfco_gradW


def compute_scfo_gH(data, W, H, kernel):
    K, T = H.shape

    # smooth data
    smooth_data = _smooth(data, kernel)

    not_eye = np.ones((K, K)) - np.eye(K)

    # apply transpose convolution
    return not_eye.dot(tensor_transconv(W, smooth_data))


def compute_smooth_kernel(maxlag):
    # TODO check
    # 2*maxlag + 1
    return np.ones([1, maxlag])


def _smooth(X, kernel):
    return convolve2d(X, kernel, 'same')
