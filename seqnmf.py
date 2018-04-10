
"""
 USAGE:

 [W, H, cost, loadings, power] = seqNMF(X, ...     X is the data matrix
       'K', 10, 'L', 20, 'lambda', .1, ...         Other inputs optional
       'W_init', W_init, 'H_init', H_init, ...
       'showPlot', 1, 'maxiter', 20, 'tolerance', -Inf, 'shift', 1, ...
       'competeW', 0, 'competeH', 0, 'lambdaL1W', 0, 'lambdaL1H', 0, ...
       'lambdaOrthoH', 0, 'lambdaOrthoW', 0)

 ------------------------------------------------------------------------------
 DESCRIPTION:

   Factorizes the NxT data matrix X into K factors
   Factor exemplars are returned in the NxKxL tensor W
   Factor timecourses are returned in the KxT matrix H

                                    ----------
                                L  /         /|
                                  /         / |
        ----------------         /---------/  |          ----------------
        |              |         |         |  |          |              |
      N |      X       |   =   N |    W    |  /   (*)  K |      H       |
        |              |         |         | /           |              |
        ----------------         /----------/            ----------------
               T                      K                         T
 See paper:
   XXXXXXXXXXXXXXXXX

 ------------------------------------------------------------------------------
 CREDITS:
   Based off of the MATLAB code provided by Emily Mackevicius and Andrew
   Bahle, 2/1/2018. Python translation by Anthony Degleris and Alex Williams,
   4/6/2018.

   Original CNMF algorithm: Paris Smaragdis 2004
   (https://link.springer.com/chapter/10.1007/978-3-540-30110-3_63)
   Adapted from NMF toolbox by Colin Vaz 2015 (http://sail.usc.edu)

   Please cite our paper:
   https://www.biorxiv.org/content/early/2018/03/02/273128
"""

import numpy as np
import numpy.linalg as la
from copy import deepcopy
from scipy.signal import convolve2d
from tqdm import trange


DEFAULT_OPTIONS = {
    'K': 10,
    'L': 100,
    'lam': 0.0,
    'W_init': None,
    'H_init': None,
    'show_plot': False,
    'maxiter': 100,
    'tol': -np.inf,
    'shift': False,
    'lamL1W': 0,
    'lamL1H': 0,
    'W_fixed': False,
    'sort_factors': True,
    'lam_orthoH': 0,
    'lam_orthoW': 0,
    'useWupdate': True
}
EPSILON = np.finfo(np.float32).eps

def seq_nmf(X, **kwargs):
    """
    INPUTS:
    ---
      X : ndarray
          Data matrix (NxT) to factorize
     K : int (optional, default value, 10)
         Number of factors

    #TODO

    L               100                                 Length (timebins) of each factor exemplar
    lam             .001                                Regularization parameter
    W_init          max(X(:))*rand(N,K,L)               Initial W
    H_init          max(X(:))*rand(K,T)./(sqrt(T/3))    Initial H (rows have norm ~1 if max(data) is 1)
    show_plot       True                                Plot every iteration? no=0
    maxiter         100                                 Maximum # iterations to run
    tol             -np.inf                             Stop if improved less than this;  Set to -inf to always run maxiter
    shift           1                                   Shift factors to center; Helps avoid local minima
    lamL1W          0                                   L1 sparsity parameter; Increase to make W's more sparse
    lamL1H          0                                   L1 sparsity parameter; Increase to make H's more sparse
    W_fixed         False                               Fix W during the fitting proceedure   
    sort_factors    1                                   Sort factors by loadings
    lam_orthoH      0                                   ||HSH^T||_1,i~=j; Encourages events-based factorizations
    lam_orthoW      0                                   ||Wflat^TWflat||_1,i~=j; ; Encourages parts-based factorizations
    useWupdate      1                                   Wupdate for cross orthogonality often doesn't change results much, and can be slow, so option to remove  
    ------------------------------------------------------------------------
     OUTPUTS:

     W                         NxKxL tensor containing factor exemplars
     H                         KxT matrix containing factor timecourses
     cost_data                 1x(#Iterations+1) vector containing 
                                   reconstruction error at each iteration. 
                                   cost(1) is error before 1st iteration.
     loadings                  1xK vector containing loading of each factor 
                                   (Fraction power in data explained by each factor)
     power                     Fraction power in data explained 
                                   by whole reconstruction
    """
    X, N, T, K, L, params = _parse_params(X, kwargs)  # Parse input
    W = params['W_init'].copy()  # Initialize
    H = params['H_init'].copy()
    Xhat = _reconstruct(W, H)
    smooth_kernel = np.ones([1, 2*L-1])  # TODO check this (row vector)
    small_num = np.max(X) * (10**-6)
    last_time = False

    cost_data = [_calc_cost(X, Xhat)]  # Calculate initial cost

    for it in trange(params['maxiter']):
        # Stopping criteria
        cost_change = cost_data[it] - np.mean(cost_data[it-5:it])
        if ((it == params['maxiter'] - 1)
           or ((it > 5) and cost_change <= params['tol'])):
            # Reached max iteration or below tolerance
            last_time = True
            if (it != 0):
                params['lambda'] = 0  # Prioritize reconstruction

        # Update H
        H = _updateH(X, N, T, K, L, params, W, H, Xhat, smooth_kernel)

        # Shift to center factors
        if (params['shift']):
            W, H = _shift_factors(W, H)
            W = W + small_num  # Add a small number to shifted W's

        # Renormalize rows of H to have constant energy
        norms = la.norm(H, axis=1)
        H = np.diag(np.divide(1, norms+EPSILON)).dot(H)
        for l in range(L):
            W[:, :, l] = W[:, :, l].dot(np.diag(norms))

        # Update W
        if(not params['W_fixed']):
            W = _updateW(X, N, T, K, L, params, W, H, smooth_kernel)

        Xhat = _reconstruct(W, H)  # Calculate cost for this iteration
        cost_data += [_calc_cost(X, Xhat)]

        if (params['show_plot']):  # Plot to show progress
            _simple_plot(W, H, Xhat, 0)

    print(it)

    # Undo zeropadding
    X = X[:, L:-L]
    Xhat = Xhat[:, L:-L]
    H = H[:, L:-L]

    # Compute explained power of reconstruction and each factor
    power = (la.norm(X)**2 - la.norm(X-Xhat)**2) / la.norm(X)**2
    loadings = _compute_loadings(X, W, H)

    # Sort factors by loading power
    if (params['sort_factors']):
        ind = np.argsort(loadings)[::-1]
        W = W[:, ind, :]
        H = H[ind, :]
        loadings = [loadings[i] for i in ind]

    return W, H, cost_data, loadings, power


def _parse_params(X, kwargs):
    """
    Parse the user's keyword arguments, setting unspecified parameters to
    their default values.
    """
    params = deepcopy(DEFAULT_OPTIONS)  # Deepcopy to prevent modifying defaults 

    for keyword in kwargs:  # Replace defaults with user parameters.  
        if (keyword not in params):
            raise ValueError('Parameter \'' + keyword + '\' not recognized.')
        params[keyword] = kwargs[keyword]
        # TODO add type detection
    K = params['K']
    L = params['L']
    N, T = X.shape

    # Initialize W_init and H_init, if not provided
    if (type(params['W_init']) != np.ndarray):
        params['W_init'] = np.max(X) * np.random.rand(N, K, L)
    if (type(params['H_init']) != np.ndarray):
        params['H_init'] = (np.max(X) / np.sqrt(T / 3.)) * np.random.rand(K, T)

    # Zeropad data by L
    X = _zero_pad(X, L)
    params['H_init'] = _zero_pad(params['H_init'], L)  # TODO check this with Alex
    N, T = X.shape  # Padded shape

    return X, N, T, K, L, params


def _zero_pad(A, L):
    """
    Adds L columns of zeros to each side of the matrix A.
    """
    N, T = A.shape
    return np.block([np.zeros([N, L]), A, np.zeros([N, L])])


def _not_eye(K):
    """
    Returns a KxK matrix with 0's along the main diagonal and 1's elsewhere.
    """
    return -np.eye(K) + 1


def _reconstruct(W, H):
    """
    INPUTS
    ---
    W : ndarray
        An NxKxL tensor which gives the neuron basis functions which are used
        for reconstructions. The L'th NxK slice of W is the nerual basis set 
        for the lag of L.
    H : ndarray
        A KxT matrix which gives timecourses for each factor.
    ---
    OUTPUTS
    ---
    Xhat : ndarray
        The reconstructed NxT matrix given by Xhat = W (*) H
    """
    N, K, L = W.shape
    K, T = H.shape

    # Zeropad by L
    H = _zero_pad(H, L)
    Xhat = np.zeros([N,T+2*L])

    for tau in range(L):  # Convolve
        Xhat = Xhat + W[:,:,tau].dot(np.roll(H, [0, tau]))

    return Xhat[:, L:-L]

def _calc_cost(X, Xhat):
    """
    Calculate the root-mean square cost error of the approximation.
    # TODO check if this is the same as the matlab cost function
    """
    return la.norm(X - Xhat) / np.sqrt(X.size)


def _updateH(X, N, T, K, L, params, W, H, Xhat, smooth_kernel):
    """
    Update H.
    """
    # Compute terms for CNMF H update
    WTX = np.zeros([K, T])
    WTXhat = np.zeros([K, T])

    for l in range(L):
        X_shifted = np.roll(X, [0, -l])
        WTX = WTX + W[:, :, l].T.dot(X_shifted)

        Xhat_shifted = np.roll(Xhat, [0, -l])
        WTXhat = WTXhat + W[:, :, l].T.dot(Xhat_shifted)

    # Compute regularization terms for H update
    dRdH = 0
    dHHdH = 0
    if (params['lam'] > 0):
        dRdH = _not_eye(K).dot(convolve2d(WTX, smooth_kernel, 'same'))
        dRdH = dRdH * params['lam']
    if (params['lam_orthoH'] > 0):
        dHHdH = _not_eye(K).dot(convolve2d(H, smooth_kernel, 'same'))
        dHHdH = dHHdH * params['lam_orthoH']

    # Update H
    num = np.multiply(H, WTX)
    denom = WTXhat + dRdH + dHHdH + params['lamL1H'] + EPSILON
    return np.divide(num, denom)


def _updateW(X, N, T, K, L, params, W, H, smooth_kernel):
    """
    Update W.
    """
    Wnew = np.zeros(W.shape)
    # Update each W[:,:,l] separately
    Xhat = _reconstruct(W, H)
    if (params['lam'] > 0 and params['useWupdate']):
        XS = convolve2d(X, smooth_kernel, 'same')
    if (params['lam_orthoW'] > 0):
        Wflat = np.sum(W, axis=2)

    for l in range(L):  # TODO parallelize?
        # Compute the terms for the CNMF W update
        H_shifted = np.roll(H, [0,l])
        XHT = X.dot(H_shifted.T)
        XhatHT = Xhat.dot(H_shifted.T)

        # Compute regularization terms for W update
        dRdW = 0
        dWWdW = 0
        if (params['lam'] > 0 and params['useWupdate']):
            dRdW = params['lam'] * XS.dot(H_shifted.T).dot(_not_eye(K))
        if (params['lam_orthoW'] > 0):
            dWWdW = params['lam_orthoW'] * Wflat.dot(_not_eye(K))
        dRdW = dRdW + dWWdW + params['lamL1W']  # Include L1 sparsity

        # Update W
        num = np.multiply(W[:,:,l], XHT); denom = XhatHT + dRdW + EPSILON
        Wnew[:,:,l] = np.divide(num, denom)
    return Wnew


def _compute_loadings(X, W, H):
    """
    Compute the fraction of power explained by each factor.
    """
    loadings = []; K,T = H.shape

    varx = la.norm(X)**2
    for i in range(K):
        WH = _reconstruct(W[:,i:i+1,:], H[i:i+1,:])
        loadings += [np.sum(2*np.multiply(X, WH) - np.power(WH, 2)) / varx]

    return loadings


def _shift_factors(W, H):
    """
    Shift factors by center of mass.
    """
    raise "Not yet implemented."
    N,K,L = W.shape; K,T = H.shape
    if (L == 1):  # No room to shift
        return W, H

    center = int(np.max([np.floor(L / 2.), 1]))

    # Pad with zeros for shifting
    Wpad = np.block([np.zeros([N,K,L]), W, np.zeros([N,K,L])])

    for k in range(K):
        temp = np.sum(W[:,k,:], axis=0)
        #if (np.sum(temp) < 10**(-10)):
        #    print(temp)
        #    raise "Problem here."

        ind = np.linspace(1, len(temp), num=len(temp), endpoint=True)
        cmass = int(np.max(np.floor(np.sum(temp.dot(ind) / (np.sum(temp) + EPSILON) ))))

        Wpad[:,k,:] = np.roll(Wpad[:,k,:], [0, center-cmass])
        H[k,:] = np.roll(H[k,:], [0, cmass-center])

    W = Wpad[:,:,L:-L]  

    return W, H

def _simple_plot(W, H, Xhat, n):
    raise "Not yet implemented."