import numpy as np  # Linear algebra
import numpy.linalg as la

from ..conv import tensor_conv, shift_and_stack, hunfold

# TODO make EPSILON universal
EPSILON = np.finfo(np.float).eps


def chals_step(data, model):
    for k in range(model.n_components):
        update_W_component(model.W, model.H, data, k)
    update_H(model.W, model.H, data)


def update_W():
    pass


def update_H(W, H, X):
    L, N, K = W.shape
    T = H.shape[1]

    W_norms = la.norm(W, axis=(0, 1))  # Compute norms of each component

    resid = X - tensor_conv(W, H)

    # Set up Wk and residual
    for k in range(K):
        Wk = W[:, :, k].T  # TODO: will this slow things down?

        # Update each entry in the row, from left to right
        for t in range(T):
            resid_slice = resid[:, t:t+L] + H[k, t] * Wk[:, :T-t]
            H[k, t] = new_H_entry(Wk[:, :T-t], W_norms[k], resid_slice)
            resid[:, t:t+L] = resid_slice - H[k, t] * Wk[:, :T-t]


def new_W_entry():
    pass


def new_H_entry(Wk, norm_Wk, resid_slice):
    """
    """
    trace = np.dot(np.ravel(Wk), np.ravel(resid_slice))

    return np.maximum(trace / (norm_Wk**2 + EPSILON), 0)


"""
"""


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
        resid_slice = resid[:, t:t+L] + H[k, t] * Wk[:, :T-t]
        H[k, t] = new_H_entry(Wk[:, :T-t], norm_Wk, resid_slice)
        resid[:, t:t+L] = resid_slice - H[k, t] * Wk[:, :T-t]


def update_W_component(W, H, X, k):
    """
    """
    L, N, K = W.shape

    W_unfold = hunfold(W)
    H_unfold = shift_and_stack(H, L)

    for l in range(L):
        ind = k*L + l
        W[l, :, k] = old_W_col_update(W_unfold, H_unfold, X, ind)


def old_W_col_update(W_unfold, H_unfold, X, ind):
    """
    Updates `W[l, :, k]` using the HALS update rule
    """

    W_col = np.dot(X, H_unfold[ind, :].T)
    for j in range(W_unfold.shape[1]):
        if (j != ind):
            W_col -= W_unfold[:, j] * np.dot(H_unfold[j, :], H_unfold[ind, :])

    W_col /= (la.norm(H_unfold[ind, :])**2 + EPSILON)

    return np.maximum(0, W_col)
