import numpy as np

from numpy.linalg import norm
from scipy.signal import convolve2d
from cnmfpy.conv import ShiftMatrix, tensor_transconv

def compute_scfo_reg(data, W, H, shifts, kernel):
    # smooth H
    maxlag = int((len(shifts)-1)/2)
    smooth_H = _smooth(H.shift(0).T, kernel)

    # penalize H
    pen_H = ShiftMatrix(np.dot(data.shift(0), smooth_H), maxlag)

    # penalize W
    penalty = tensor_transconv(W, pen_H, shifts)
    return norm(penalty)


def compute_scfo_gW(data, W, H, shifts, kernel):
    K, T = H.shape

    # preallocate
    sfco_gradW = np.empty(W.shape)

    #smooth H
    smooth_H = _smooth(H.shift(0).T, kernel)

    not_eye = np.ones((K, K)) - np.eye(K)

    # TODO: broadcast
    for l, t in enumerate(shifts):
        sfco_gradW[l] = data.shift(-t).dot(smooth_H).dot(not_eye)

    return sfco_gradW


def compute_scfo_gH(data, W, H, shifts, kernel):
    K, T = H.shape

    # smooth data
    maxlag = int((len(shifts)-1)/2)
    smooth_data = ShiftMatrix(_smooth(data.shift(0), kernel), maxlag)

    not_eye = np.ones((K, K)) - np.eye(K)

    # apply transpose convolution
    return not_eye.dot(tensor_transconv(W, smooth_data, shifts))

def compute_smooth_kernel(maxlag):
    # TODO check
    return np.ones([1, 4*maxlag+1])


def _smooth(X, kernel):
    return convolve2d(X, kernel, 'same')