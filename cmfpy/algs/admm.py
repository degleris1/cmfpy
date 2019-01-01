import numpy as np
import numpy.random as npr
import numpy.linalg as la

from ..common import cmf_predict
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
        super().__init__(data, dims, patience=patience, tol=tol, **kwargs)

        self.Q = self.W
        self.R = self.H

        self.Lambda = np.zeros_like(self.W)
        self.Pi = np.zeros_like(self.H)
        self.Theta = np.zeros_like(self.H)

    def update(self):
        self.update_W()
        self.update_H()
        self.update_Q()
        self.update_R()
        
        self.update_Lambda()
        self.update_Pi()
        self.update_Theta()

        self.cache_resids()
        return self.loss

    def update_W(self):
        pass

    def update_H(self):
        pass

    def update_Q(self):
        pass

    def update_R(self):
        pass

    def update_Lambda(self):
        pass

    def update_Pi(self):
        pass

    def update_Theta(self):
        pass

    def rand_init(self):
        """Overrides initialization so that W and H are the right shape for ADMM"""

        # Randomize nonnegative parameters.
        W = npr.rand(self.n_components, self.n_features, self.maxlag)
        H = npr.rand(self.n_components, self.maxlag, self.n_timepoints)

        # Correct scale of parameters, rescaling each component of H individually.
        W_reshaped = np.swapaxes(W, 0, 2)

        for i in range(self.maxlag):
            H_reshaped = H[:,i,:]
            est = cmf_predict(W_reshaped, H_reshaped)
            alpha = (self.X * est).sum() / la.norm(est)**2
            H[:,i,:] = alpha * H[:,i,:]

        return alpha * W, alpha * H

    def cache_resids(self):
        """
        Update residuals. 

        Since W and H are not always guaranteed to be feasible, we instead use
        Q and R (which are always feasible) to calculate the residual.
        """

        # When cache_resids is called by the superclass initializer, 
        # Q and R will not yet be initialized, in that case use W and H instead.
        if ((self.Q is None) or (self.R is None)):
            Q = np.swapaxes(self.W, 0, 2)
            R = self.H[:,0,:]
        else: 
            Q = np.swapaxes(self.Q, 0, 2)
            R = self.R[:,0,:]

        self.est = cmf_predict(Q, R)
        self.resids = self.est - self.X


        