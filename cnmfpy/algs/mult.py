import numpy as np 
from tqdm import trange

from cnmfpy.conv import ShiftMatrix, tensor_transconv
from cnmfpy.optimize import compute_loss, renormalize, shift_factors
from cnmfpy.regularize import compute_scfo_gW, compute_scfo_gH


EPSILON = np.finfo(np.float32).eps


def fit_mult(data, model):
    m, n = data.shape
    shifts = model._shifts

    # initial loss
    model.loss_hist = [compute_loss(data, model.W, model.H, shifts)]

    converged, itr = False, 0
    for itr in trange(model.n_iter_max):
        # shift factors
        #if ((itr % 5 == 0) and (model.n_iter_max - itr > 15)):
            #model.W, model.H = shift_factors(model.W, model.H, shifts)

        # compute multiplier for W
        num_W, denom_W = _compute_mult_W(data, model)

        # update W
        model.W = np.divide(np.multiply(model.W, num_W), denom_W+EPSILON)
        #model.loss_hist += [compute_loss(data, model.W, model.H, shifts)]

        # compute multiplier for H
        num_H, denom_H = _compute_mult_H(data, model)

        # update H
        model.H.assign(np.divide(np.multiply(model.H.shift(0), num_H), denom_H+EPSILON))
        model.loss_hist += [compute_loss(data, model.W, model.H, shifts)]

        # renormalize H
        model.W, model.H = renormalize(model.W, model.H)

        # check convergence
        prev_loss, new_loss = model.loss_hist[-2:]
        if (np.abs(prev_loss - new_loss) < model.tol):
            converged = True
            break


def _compute_mult_W(data, model):
    # preallocate
    #mult_W = np.zeros(model.W.shape)
    num = np.zeros(model.W.shape)
    denom = np.zeros(model.W.shape)

    est = model.predict()
    reg_gW = model.l2_scfo * compute_scfo_gW(data, model.W, model.H, 
                                            model._shifts, model._kernel)

    # TODO: broadcast
    for l, t in enumerate(model._shifts):
        num[l] = np.dot(data.shift(0), model.H.shift(t).T)
        denom[l] = np.dot(est, model.H.shift(t).T) + reg_gW[l] + model.l1_W
        #mult_W[l] = np.divide(num, denom + EPSILON)

    return num, denom  #mult_W


def _compute_mult_H(data, model):

    est = ShiftMatrix(model.predict(), model.maxlag)
    reg_gH = model.l2_scfo * compute_scfo_gH(data, model.W, model.H,
                                            model._shifts, model._kernel)

    num = tensor_transconv(model.W, data, model._shifts)
    denom = tensor_transconv(model.W, est, model._shifts) + reg_gH + model.l1_H
   
    return num, denom  #np.divide(num, denom + EPSILON)