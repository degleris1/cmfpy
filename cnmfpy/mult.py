import numpy as np 
from tqdm import trange

from conv import ShiftMatrix, tensor_transconv
from optimize import soft_thresh, compute_loss


EPSILON = np.finfo(np.float32).eps


def fit_mult(data, model):
    m, n = data.shape
    shifts = model._shifts


    # initial loss
    model.loss_hist = [compute_loss(data, model.W, model.H, shifts)]

    converged, itr = False, 0
    for itr in trange(model.n_iter_max):
        # compute multiplier for W
        mult_W = _compute_mult_W(data, model)

        # update W
        model.W = soft_thresh(np.multiply(model.W, mult_W), model.l1_W)
        model.loss_hist += [compute_loss(data, model.W, model.H, shifts)]

        # compute multiplier for H
        mult_H = _compute_mult_H(data, model)

        # update H
        model.H.assign(soft_thresh(np.multiply(model.H.shift(0), mult_H), model.l1_H))
        model.loss_hist += [compute_loss(data, model.W, model.H, shifts)]

        # check convergence
        prev_loss, new_loss = model.loss_hist[-2:]
        if (np.abs(prev_loss - new_loss) < model.tol):
            converged = True
            break


def _compute_mult_W(data, model):
    # preallocate
    mult_W = np.zeros(model.W.shape)

    est = model.predict()

    # TODO: broadcast
    for l, t in enumerate(model._shifts):
        num = np.dot(data.shift(0), model.H.shift(t).T)
        denom = np.dot(est, model.H.shift(t).T)
        mult_W[l] = np.divide(num, denom + EPSILON)

    return mult_W


def _compute_mult_H(data, model):

    est = ShiftMatrix(model.predict(), model.maxlag)

    num = tensor_transconv(model.W, data, model._shifts)
    denom = tensor_transconv(model.W, est, model._shifts)
   
    return np.divide(num, denom + EPSILON)