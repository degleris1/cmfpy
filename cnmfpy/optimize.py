import numpy as np
import numba

from numpy.linalg import norm
from cnmfpy.conv import ShiftMatrix, tensor_conv, tensor_transconv


EPSILON = np.finfo(np.float32).eps


def compute_gH(data, W, H):
    """
    Compute the gradient of H.
    """
    # compute estimate
    est = tensor_conv(W, H)

    # compute residual and loss
    resid = est - data.shift(0)
    loss = norm(resid)

    # wrap residual in ShiftMatrix
    resid = ShiftMatrix(resid, W.shape[0])

    # compute grad
    Hgrad = tensor_transconv(W, resid)

    return loss, Hgrad


def compute_gW(data, W, H):
    """
    Compute the gradient of W.
    """
    # compute estimate
    est = tensor_conv(W, H)

    # compute residual and loss
    resid = est - data.shift(0)
    loss = norm(resid)

    # TODO: replace with broadcasting
    Wgrad = np.empty(W.shape)

    for l in np.arange(W.shape[0]):
        Wgrad[l] = np.dot(resid, H.shift(l).T)

    return loss, Wgrad


def soft_thresh(X, l):
    """
    Soft thresholding function for sparsity.
    """
    return np.maximum(X-l, 0) - np.maximum(-X-l, 0)


def compute_loss(data, W, H):
    """
    Compute the loss of a CNMF factorization.
    """
    resid = tensor_conv(W, H) - data.shift(0)
    return norm(resid)


def compute_loadings(data, W, H):
    """
    Compute the power explained by each factor.
    """
    loadings = []
    K, T = H.shape
    maxlag = W.shape[0]

    data_mag = norm(data.shift(0))

    for i in range(K):
        Wi = W[:, :, i:i+1]
        Hi = ShiftMatrix(H.shift(0)[i:i+1, :], maxlag)
        est = tensor_conv(Wi, Hi)
        loadings += [norm(est - data.shift(0)) / (data_mag + EPSILON)]

    return loadings


def renormalize(W, H):
    """ Renormalizes the rows of H to have constant energy.
    Updates passed parameters.
    """
    L, N, K = W.shape

    row_norms = norm(H.shift(0), axis=1)
    H.assign(np.diag(np.divide(1, row_norms+EPSILON)).dot(H.shift(0)))

    for l in range(L):
        W[l] = W[l].dot(np.diag(row_norms+EPSILON))

    return W, H


def shift_factors(W, H):
    """Shift factors by their center of mass."""
    maxlag, N, K = W.shape

    if (L == 1):
        raise IndexError('No room to shift. Disable shifting.')

    center = int(np.floor(L / 2))
    #Wpad = np.pad(W, ((0, 0), (L, L), (0, 0)))

    # TODO broadcast
    shifted_H = np.zeros(H.shift(0).shape)
    for k in range(K):
        masses = np.sum(W[:,k,:], axis=1)

        if (np.sum(masses) > EPSILON):
            ind = np.arange(0, L)
            cmass = int(np.floor(np.dot(masses, ind) / np.sum(masses)))+1
            l = center - cmass

            Wpad = np.pad(W[:, :, k], ((L, L),(0, 0)), mode='constant')
            W[:, :, k] = Wpad[L - l: 2*L - l,:]

            shifted_H[k, :] = H.shift(-l)[k, :]

    H.assign(shifted_H)

    return W, H