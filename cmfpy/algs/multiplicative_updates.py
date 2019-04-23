"""
Multiplicative Update algorithm.
"""
import numba
import numpy as np
from ..common import tconv, t_tconv, sdot, t_sdot

EPSILON = 1e-7


# @numba.jit(nopython=True)
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

    WtW = np.empty((L * L, K, K))
    W_nums = np.empty((L, N, K))
    W_denom = np.empty((N, K))

    # Control iterations.
    last_W = np.empty_like(W)
    last_H = np.empty_like(H)
    itercount = 0
    converged = False
    stop_early = False

    # == MAIN LOOP == #

    while (itercount < max_iter) and (not converged):

        # == COPY PARAMS FOR CONVERGENCE CHECK == #
        np.copyto(last_W, W)
        np.copyto(last_H, H)

        # == UPDATE W == #

        # Precompute gram matrices for all lags
        for lag in range(-L + 1, L):
            HHt[lag] = t_sdot(H, H, lag)

        for lag in range(L):
            W_nums[lag] = t_sdot(X, H, lag)

        # Update W multiple times (accelerated MU)
        for s in range(acc_steps):

            # Update each slice of W
            for i in range(L):

                # Compute denominator.
                W_denom.fill(EPSILON)
                for j in range(L):
                    W_denom += np.dot(W[j], HHt[j - i + L - 1])

                # Update parameters.
                W[i] *= W_nums[i] / W_denom

        # == UPDATE H == #

        # Compute numerator (transposed tensor convolution.)
        H_num.fill(0.0)
        for lag, Wl in enumerate(W):
            H_num += sdot(Wl.T, X, -lag)

        # Compute WtW gramians for denominator. (All combinations of lags.)
        i, j = 0, 0
        for k in range(L * L):
            WtW[k] = np.dot(W[i].T, W[j])
            j = (j + 1) % L
            i = i if j else (i + 1)

        # Update H multiple times (accelerated MU)
        for s in range(acc_steps):

            # Compute denominator, iterating over all lags.
            H_denom.fill(EPSILON)

            i, j = 0, 0
            for k in range(L * L):

                H_denom += sdot(WtW[k], H, j - i)
                j = (j + 1) % L
                i = i if j else (i + 1)

            # Update parameters
            H *= H_num / H_denom

        # == CHECK CONVERGENCE == #
        itercount += 1
        H_crit = np.ptp(last_H - H) / np.ptp(H)
        W_crit = np.ptp(last_W - W) / np.ptp(W)
        converged = (H_crit < tol) and (W_crit < tol)

    return W, H, converged
