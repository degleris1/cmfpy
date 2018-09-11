import numpy as np
from tqdm import trange

from numpy.linalg import norm

from cnmfpy.conv import tensor_transconv, shift_cols
from cnmfpy.optimize import compute_loss, renormalize
from cnmfpy.regularize import compute_scfo_gW, compute_scfo_gH


# TODO float or float32?
EPSILON = np.finfo(np.float).eps


def fit_mult(data, model):
    m, n = data.shape

    # initial loss
    model.loss_hist = [compute_loss(data, model.W, model.H)]

    itr = 0
    for itr in trange(model.n_iter_max):
        print('W', np.sum(model.W), ', H', np.sum(model.H))

        if (np.isnan(model.W).any()):
            raise Exception('W has Nans!!')

        if (np.isnan(model.H).any()):
            raise Exception('H has NANs!!')

        # shift factors
        # if ((itr % 5 == 0) and (model.n_iter_max - itr > 15)):
            # model.W, model.H = shift_factors(model.W, model.H)

        # compute multiplier for W
        num_W, denom_W = _compute_mult_W(data, model)

        # update W
        model.W = np.divide(np.multiply(model.W, num_W), denom_W)

        # compute multiplier for H
        num_H, denom_H = _compute_mult_H(data, model)

        # update H
        model.H = np.divide(np.multiply(model.H, num_H), denom_H)
        model.loss_hist += [compute_loss(data, model.W, model.H)]

        # renormalize H
        model.W, model.H = renormalize(model.W, model.H)

        # check convergence
        prev_loss, new_loss = model.loss_hist[-2:]
        if (np.abs(prev_loss - new_loss) < model.tol):
            break


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

    #print('W', norm(num), norm(denom))

    return num, denom + EPSILON


def _compute_mult_H(data, model):
    est = model.predict()
    reg_gH = model.l2_scfo * compute_scfo_gH(data, model.W, model.H,
                                             model._kernel)

    num = tensor_transconv(model.W, data)
    denom = tensor_transconv(model.W, est) + reg_gH + model.l1_H

    #print(norm(np.divide(num, denom)))

    #print('H', norm(num), norm(denom))

    return num, denom + EPSILON
