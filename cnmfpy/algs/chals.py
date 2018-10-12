import numpy as np  # Linear algebra
import numpy.linalg as la

from tqdm import trange

from ..conv import tensor_conv, shift_and_stack, hunfold
from ..optimize import compute_loss, renormalize

# TODO make EPSILON universal
EPSILON = np.finfo(np.float).eps


def fit_chals(data, model):
    N, T = data.shape

    # initial loss
    model.loss_hist = [compute_loss(data, model.W, model.H)]

    itr = 0
    for itr in trange(model.n_iter_max):

        if (np.isnan(model.W).any()):
            raise Exception('W has Nans!!')

        if (np.isnan(model.H).any()):
            raise Exception('H has NANs!!')

        # shift factors
        # if ((itr % 5 == 0) and (model.n_iter_max - itr > 15)):
            # model.W, model.H = shift_factors(model.W, model.H)

        for k in range(model.n_components):
            update_W_component(model.W, model.H, data, k)
            update_H_component(model.W, model.H, data, k)

        # Update loss history and renormalize H
        model.loss_hist += [compute_loss(data, model.W, model.H)]
        model.W, model.H = renormalize(model.W, model.H)

        # check convergence
        prev_loss, new_loss = model.loss_hist[-2:]
        if (np.abs(prev_loss - new_loss) < model.tol):
            break


def update_W_col(W, H, X, k, l):
    """
    Updates `W[l, :, k]` using the HALS update rule
    """
    L, N, K = W.shape

    W_unfold = hunfold(W)
    H_unfold = shift_and_stack(H, L)

    ind = k*L + l

    W_col = np.dot(X, H_unfold[ind, :].T)
    for j in range(W_unfold.shape[1]):
        if (j != ind):
            W_col -= W_unfold[:, j] * np.dot(H_unfold[j, :], H_unfold[ind, :])

    W_col /= (la.norm(H_unfold[ind, :])**2 + EPSILON)

    # DEBUG
    # print(W_col, '\n')

    W[l, :, k] = np.maximum(0, W_col)


def update_W_component(W, H, X, k):
    """
    """
    L, N, K = W.shape

    for l in range(L):
        update_W_col(W, H, X, k, l)


def new_H_entry(Wk, norm_Wk, resid_slice):
    """
    """
    trace = np.dot(np.ravel(Wk), np.ravel(resid_slice))

    return np.maximum(trace / (norm_Wk**2 + EPSILON), 0)


def update_H_component(W, H, X, k):
    """
    """
    L, N, K = W.shape
    T = H.shape[1]

    # Set up Wk and residual
    Wk = W[:, :, k].T  # TODO: will this slow things down?
    norm_Wk = la.norm(Wk)
    resid = X - tensor_conv(W, H)

    # Update each entry in the row, from left to right
    for t in range(T):
        resid_slice = resid[:, t:t+L] + (H[k, t] * Wk)[:, :T-t]
        H[k, t] = new_H_entry(Wk[:, :T-t], norm_Wk, resid_slice)
