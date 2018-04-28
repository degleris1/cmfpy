import numpy as np
import numpy.linalg as la
from tqdm import trange

from conv import ShiftMatrix
from optimize import compute_gW, compute_gH, soft_thresh, compute_loss


def fit_bcd(data, model):

    m, n = data.shape
    shifts = model._shifts

    converged, itr = False, 0
    model.loss_hist = []
    
    for itr in trange(model.n_iter_max):
        
        # compute gradient and step size of W
        loss_1, grad_W = compute_gW(data, model.W, model.H, shifts)
        step_W = _scale_gW(data, grad_W, model)

        # update W
        new_W = np.maximum(model.W - np.multiply(step_W, grad_W), 0)
        model.W = soft_thresh(new_W, model.l1_W)

        # compute gradient and step size of H
        loss_2, grad_H = compute_gH(data, model.W, model.H, shifts)
        step_H = _scale_gH(data, grad_H, model)

        # update H
        new_H = np.maximum(model.H.shift(0) - np.multiply(step_H, grad_H), 0)
        model.H.assign(soft_thresh(new_H, model.l1_H))

        model.loss_hist += [loss_1, loss_2]

        # check convergence
        prev_loss, new_loss = model.loss_hist[-2:]
        if (np.abs(prev_loss - new_loss) < model.tol):
            converged = True
            break



def _scale_gW(data, grad_W, model, step_type='backtrack'):
    if (step_type == 'backtrack'):
        step_W = _backtrack(data, grad_W, 0, model)

    else:
        raise ValueError('Invalid BCD step type.')

    return step_W


def _scale_gH(data, grad_H, model, step_type='backtrack'):
    if (step_type == 'backtrack'):
        step_H = _backtrack(data, 0, grad_H, model)

    else:
        raise ValueError('Invalid BCD step type.')

    return step_H


def _backtrack(data, grad_W, grad_H, model, beta=0.8, alpha=0.00001, 
                max_iters=500):
    """
    Backtracking line search to find a step length.
    """
    # compute initial loss and gradient magnitude
    shifts = model._shifts
    past_loss = compute_loss(data, model.W, model.H, shifts)
    grad_mag = la.norm(grad_W)**2 + la.norm(grad_H)**2

    # first check
    t = 1.0
    new_H = ShiftMatrix(np.maximum(model.H.shift(0) - t*grad_H, 0), model.maxlag)
    new_W = np.maximum(model.W - t*grad_W, 0)
    new_loss = compute_loss(data, new_W, new_H, shifts)

    iters = 0
    # backtracking line search
    while ((new_loss > past_loss - alpha*t*grad_mag) and (iters < max_iters)):
        t = beta * t

        new_H.assign(np.maximum(model.H.shift(0) - t*grad_H, 0))
        new_W = np.maximum(model.W - t*grad_W, 0)
        new_loss = compute_loss(data, new_W, new_H, shifts)

        iters += 1

    return t


# TODO: compute the lipschitz constant for optimal learning rate