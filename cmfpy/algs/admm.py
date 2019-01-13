import numpy as np
import numpy.random as npr
import numpy.linalg as la

from ..common import cmf_predict, shift_and_fill
from .base import AbstractOptimizer, EPSILON

class ADMMUpdate(AbstractOptimizer):
    """
    Perform updates using ADMM (Alternating-Directions Method of Multipliers).
    This casts the problem as an equality-constrained minimization problem with
    4 primal variables and 3 dual variables. 

    We will use additional primal tensor variables Q and R, which are the
    counterparts of W and H, respectively. In this formulation, W and H are not
    constrained to be feasible at every iteration. However, Q and R are always
    nonnegative. At convergence, W = Q and H = R.

    We also introduce the dual variables Lambda, Pi, and Theta. Lambda and Pi
    are used to enforce the equality constraints W = Q and H = R. Theta is used
    to enforce the L linear matrix equality constraints so that each component
    of H is equal to the prior component shifted over.
    """

    def __init__(self, data, dims, rho=1, patience=3, tol=1e-5, **kwargs):
        """
        Initialize variables for ADMM. 
        
        We overrride the superclass because the cache_resids function is
        overridden for ADMM, and we cannot call cache_resids without first
        initializing Q and R.

        By constrast to cHALS, it makes ADMM simpler if we choose W to have
        dimensions K x N x L and H to have dimensions K x L x T. W[i,:,:] is the
        ith component of W, and H[j,:,:] is the jth component of H.
        """
        self.Q = None
        self.R = None
        self.n_components = None
        self.n_timepoints = None
        self.maxlag = None
        super().__init__(data, dims, patience=patience, tol=tol, **kwargs)

        self.rho = rho
        self.Q = self.W
        self.R = self.H

        self.Lambda = np.zeros_like(self.W)
        self.Pi = np.zeros_like(self.H)
        self.Theta = np.zeros_like(self.H)

        self.sum = np.zeros_like(self.X)
        for l in range(self.maxlag):
            self.sum += self.W[l,:,:] @ self.H[l,:,:]

        self.cache_resids()

    def update(self):
        assert(not np.any(np.isnan(self.W)))
        assert(not np.any(np.isnan(self.H)))
        
        # self.update_components(self.update_Wk)
        # self.update_components(self.update_Qk)

        # self.update_components(self.update_Hk)
        # self.update_components(self.update_Rk)
        
        # self.update_components(self.update_Lambdak)
        # self.update_components(self.update_Pik)
        # self.update_components(self.update_Thetak)
        self.gauss_seidel_update()
        self.cache_resids()
        return self.loss

    def gauss_seidel_update(self):
        for l in range(self.maxlag):
            rho = self.rho
            self.update_Wk(l)
            self.Lambda += rho * (self.W - self.Q)
            self.update_Qk(l)

            self.update_Hk(l)
            self.Pi += rho * (self.H - self.R)

            self.update_Rk(l)
            self.update_Thetak(l)

    def update_components(self, component_updater):
        # Iterate over all components, and call the componenet updater on each
        for l in range(self.maxlag):
            component_updater(l)

    def update_Wk(self, k):
        rho = self.rho
        X = self.X
        W = self.W[k,:,:]
        H = self.H[k,:,:]
        E = self.sum - (W @ H)
        Q = self.Q[k,:,:]
        Lambda = self.Lambda[k,:,:]
        
        A =  2 * X @ H.T - E @ H.T - Lambda + rho * Q
        B = (2 * H @ H.T + rho * np.identity(H.shape[0]))
        self.W[k,:,:] = A @ la.inv(B)

        # Update sum to reflect updates to Wk
        self.sum = E + self.W[k,:,:] @ H

    def update_Hk(self, k):
        rho = self.rho
        X = self.X
        W = self.W[k,:,:]
        H = self.H[k,:,:]
        H0 = self.H[0,:,:]
        E = self.sum - (W @ H)
        Q = self.Q[k,:,:]
        Pi = self.Pi[k,:,:]
        R = self.R[k,:,:]
        Theta = self.Theta[k,:,:]
        Lambda = self.Lambda[k,:,:]

        l = H.shape[0]
        if (k == 0):
            A = 2 * W.T @ W + rho * np.identity(l)
            B = 2 * W.T @ X - W.T @ E - Pi + rho * R
            self.H[k,:,:] = la.inv(A) @ B
        else:
            H0_shifted = shift_and_fill(H0, k)
            A = 2 * W.T @ W + 2 * rho * np.identity(l)
            B = 2 * W.T @ X - W.T @ E - Theta - Pi + rho * R + rho * H0_shifted
            self.H[k,:,:] = la.inv(A) @ B
        
        # update sum
        self.sum = E + W @ self.H[k,:,:]

    def update_Qk(self, k):
        rho = self.rho
        Lambda = self.Lambda[k,:,:]
        W = self.W[k,:,:]
        self.Q[k,:,:] = np.maximum(1.0 / rho * Lambda + W, 0)

    def update_Rk(self, k):
        rho = self.rho
        Pi = self.Pi[k,:,:]
        H = self.H[k,:,:]
        self.R[k,:,:] = np.maximum(1.0 / rho * Pi + H, 0)

    def update_Lambdak(self, k):
        rho = self.rho
        W = self.W[k,:,:]
        Q = self.Q[k,:,:]
        self.Lambda[k,:,:] = self.Lambda[k,:,:] + rho * (W - Q)

    def update_Pik(self, k):
        rho = self.rho
        H = self.H[k,:,:]
        R = self.R[k,:,:]
        self.Pi[k,:,:] = self.Pi[k,:,:] + rho * (H - R)

    def update_Thetak(self, k):
        rho = self.rho
        H = self.H[k,:,:]
        H0 = self.H[0,:,:]
        H0_shifted = shift_and_fill(H0, k)
        self.Theta[k,:,:] = self.Theta[k,:,:] + rho * (H - H0_shifted)

    def rand_init(self):
        """Overrides initialization so that W and H are the right shape for ADMM"""

        # Randomize nonnegative parameters.
        W = npr.rand(self.maxlag, self.n_features, self.n_components)

        # change initialization so that they're shifted copies of each other
        H0 = npr.rand(self.n_components, self.n_timepoints)
        H = np.expand_dims(H0, 0)
        for l in range(self.maxlag):
            H_next = np.expand_dims(shift_and_fill(H0,l), 0)
            H = np.concatenate((H, H_next), 0)

        est = cmf_predict(W, H0)
        alpha = (self.X * est).sum() / la.norm(est)**2
    
        return np.sqrt(alpha) * W, np.sqrt(alpha) * H

    def cache_resids(self):
        """
        Update residuals. 

        Since W and H are not always guaranteed to be feasible, we instead use
        Q and R (which are always feasible) to calculate the residual.
        """

        # When cache_resids is called by the superclass initializer, 
        # Q and R will not yet be initialized, in that case use W and H instead.
        if ((self.Q is None) or (self.R is None)):
            Q = self.W
            R = self.H[0,:,:]
        else: 
            Q = self.Q
            R = self.R[0,:,:]

        self.est = cmf_predict(Q, R)
        self.resids = self.est - self.X

def converged(self, loss_hist):
    """
    ADMM has converged when everything is primal feasible,
    and the variables stop updating.
    """
    # Improvement in loss function over iteration.
    d_loss = np.diff(loss_hist[-self.patience:])

    # Objective converged
    if (np.all(np.abs(d_loss) < self.tol)):
        loss_conv = True
    else:
        loss_conv = False

    return (la.norm(self.W - self.Q) < self.tol) and loss_conv