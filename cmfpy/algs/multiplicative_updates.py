"""
Multiplicative Update algorithm.
"""
import numba
import numpy as np
from ..common import sdot, sdot_plus, t_sdot
from .utils import param_convergence

EPSILON = 1e-7


@numba.jit(nopython=True)
def mult(
        X, W, H, l1_W=0.0, l1_H=0.0, l2_W=0.0, l2_H=0.0,
        acc_steps=5, max_iter=100, verbose=False, tol=1e-4):

    # Get dimensions.
    L, N, K = W.shape
    _, T = H.shape

    # Preallocate space.
    HHt = np.empty((2 * L - 1, K, K))
    H_num = np.empty((K, T))
    H_denom = np.empty((K, T))

    WtW = np.empty(((L * L + L) // 2, K, K))
    W_nums = np.empty((L, N, K))
    W_denom = np.empty((N, K))

    # Control iterations.
    last_W = np.empty((L, N, K))
    last_H = np.empty((K, T))
    itercount = 0
    converged = False
    stop_early = False

    # == MAIN LOOP == #

    while (itercount < max_iter) and (not converged):

        H[:, -L:] *= 0.0

        # == COPY PARAMS FOR CONVERGENCE CHECK == #
        last_W[:] = W
        last_H[:] = H

        # == UPDATE W == #

        # Precompute gram matrices for all lags
        for k, lag in enumerate(range(-L + 1, L)):
            HHt[k] = t_sdot(H, H, lag)

        for lag in range(L):
            W_nums[lag] = t_sdot(X, H, lag)

        # Update W multiple times (accelerated MU)
        for s in range(acc_steps):

            # Update each slice of W
            for i in range(L):

                # Compute denominator.
                W_denom.fill(EPSILON)
                for j in range(L):
                    W_denom += np.dot(W[j], HHt[L - 1 + j - i])

                # Update parameters.
                W[i] *= W_nums[i] / W_denom

        # == UPDATE H == #

        # Compute numerator (transposed tensor convolution.)
        H_num.fill(0.0)
        for lag in range(L):
            sdot_plus(W[lag].T, X, -lag, H_num)

        # Compute WtW gramians for denominator. (All combinations of lags.)
        i, j = 0, 0
        for k in range(len(WtW)):
            WtW[k] = np.dot(W[i].T, W[j])
            j += 1
            if j == L:
                i += 1
                j = i

        # Update H multiple times (accelerated MU)
        for s in range(acc_steps):

            # Compute denominator, iterating over all lags.
            H_denom.fill(EPSILON)

            i, j = 0, 0
            for k in range(len(WtW)):
                if i == j:
                    H_denom += np.dot(WtW[k], H)
                else:
                    sdot_plus(WtW[k], H, j - i, H_denom)
                    sdot_plus(WtW[k].T, H, i - j, H_denom)

                j += 1
                if j == L:
                    i += 1
                    j = i

            # Update parameters
            H *= H_num / H_denom

        # == CHECK CONVERGENCE == #
        itercount += 1
        H_crit = param_convergence(H.ravel(), last_H.ravel())
        W_crit = param_convergence(W.ravel(), last_W.ravel())
        converged = (H_crit < tol) and (W_crit < tol)

    info = [
        ("converged", converged),
        ("itercount", itercount),
    ]
    return W, H, info


@numba.jit(nopython=True)
def _alt_mult(
        X, W, H, l1_W=0.0, l1_H=0.0, l2_W=0.0, l2_H=0.0,
        acc_steps=1, max_iter=100, verbose=False, tol=1e-4):

    # Get dimensions.
    L, N, K = W.shape
    _, T = H.shape

    est = np.empty_like(X)
    H_num = np.empty_like(H)
    H_denom = np.empty_like(H)

    # Control iterations.
    last_W = np.empty((L, N, K))
    last_H = np.empty((K, T))
    itercount = 0
    converged = False
    stop_early = False

    # == MAIN LOOP == #

    while (itercount < max_iter) and (not converged):

        H[:, -L:] *= 0.0

        # == COPY PARAMS FOR CONVERGENCE CHECK == #
        last_W[:] = W
        last_H[:] = H

        # == UPDATE W == #

        est.fill(0.0)
        for lag in range(L):
            sdot_plus(W[lag], H, lag, est)

        for lag in range(L):
            W[lag] *= t_sdot(X, H, lag) / (EPSILON + t_sdot(est, H, lag))

        # == UPDATE H == #

        est.fill(0.0)
        for lag in range(L):
            sdot_plus(W[lag], H, lag, est)

        H_num.fill(0.0)
        H_denom.fill(EPSILON)
        for lag in range(L):
            sdot_plus(W[lag].T, X, -lag, H_num)
            sdot_plus(W[lag].T, est, -lag, H_denom)

        H *= H_num / H_denom

        # == CHECK CONVERGENCE == #
        itercount += 1
        H_crit = param_convergence(H.ravel(), last_H.ravel())
        W_crit = param_convergence(W.ravel(), last_W.ravel())
        converged = (H_crit < tol) and (W_crit < tol)

    info = [
        ("converged", converged),
        ("itercount", itercount),
    ]
    return W, H, info
