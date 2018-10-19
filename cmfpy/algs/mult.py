import numpy as np

from ..conv import tensor_transconv, shift_cols
from ..regularize import compute_scfo_gW, compute_scfo_gH


# TODO float or float32?
EPSILON = np.finfo(np.float).eps


def mult_step(data, model):
    # compute multiplier for W
    num_W, denom_W = _compute_mult_W(data, model)
    # update W
    model.W = np.divide(np.multiply(model.W, num_W), denom_W)

    # compute multiplier for H
    num_H, denom_H = _compute_mult_H(data, model)
    # update H
    model.H = np.divide(np.multiply(model.H, num_H), denom_H)


def _compute_mult_W(data, model):
    # preallocate
    num = np.zeros(model.W.shape)
    denom = np.zeros(model.W.shape)

    est = model.predict()
    reg_gW = model.l2_scfo * compute_scfo_gW(data, model.W, model.H,
                                             model._kernel)

    # TODO: broadcast
    for l in np.arange(model.W.shape[0]):
        num[l] = np.dot(data[:, l:], shift_cols(model.H, l).T)
        denom[l] = np.dot(est[:, l:], shift_cols(model.H, l).T) + \
                          reg_gW[l] + model.l1_W

    return num, denom + EPSILON


def _compute_mult_H(data, model):
    est = model.predict()
    reg_gH = model.l2_scfo * compute_scfo_gH(data, model.W, model.H,
                                             model._kernel)

    num = tensor_transconv(model.W, data)
    denom = tensor_transconv(model.W, est) + reg_gH + model.l1_H

    return num, denom + EPSILON
