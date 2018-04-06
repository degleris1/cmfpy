
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
from copy import deepcopy


DEFAULT_OPTIONS = {
	'K': 10,
	'L': 100,
	'lam': 0.001,
	'W_init': None,
	'H_init': None,
	'show_plot': True,
	'maxiter': 100,
	'tol': None,
	'shift': True,
	'lamL1W': False,
	'lamL1H': False,
	'W_fixed': False,
	'sort_factors': True,
	'lam_orthoH': False,
	'lam_orthoW': False,
	'useWupdate': True
}

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
	show_plot       1                                   Plot every iteration? no=0
	maxiter         100                                 Maximum # iterations to run
	tol             None                                Stop if improved less than this;  Set to None to always run maxiter
	shift           1                                   Shift factors to center; Helps avoid local minima
	lamL1W          0                                   L1 sparsity parameter; Increase to make W's more sparse
	lamL1H          0                                   L1 sparsity parameter; Increase to make H's more sparse
	W_fixed         0                                   Fix W during the fitting proceedure   
	sort_factors    1                                   Sort factors by loadings
	lam_orthoH      0                                   ||HSH^T||_1,i~=j; Encourages events-based factorizations
	lam_orthoW      0                                   ||Wflat^TWflat||_1,i~=j; ; Encourages parts-based factorizations
	useWupdate      1                                   Wupdate for cross orthogonality often doesn't change results much, and can be slow, so option to remove  
	------------------------------------------------------------------------
	 OUTPUTS:

	 W                         NxKxL tensor containing factor exemplars
	 H                         KxT matrix containing factor timecourses
	 cost                      1x(#Iterations+1) vector containing 
	                               reconstruction error at each iteration. 
	                               cost(1) is error before 1st iteration.
	 loadings                  1xK vector containing loading of each factor 
	                               (Fraction power in data explained by each factor)
	 power                     Fraction power in data explained 
	                               by whole reconstruction
	"""
	print('Not yet implemented!')

	X, N, T, K, L, params = parse_params(X, kwargs)  # parse input




def parse_params(X, kwargs):
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
	K = params['K']; L = params['L']

	# Zeropad data by L
	X = zero_pad(X,L)
	N, T = X.shape

	# Initialize W_init and H_init, if not provided
	if (type(params['W_init']) != np.ndarray):
		params['W_init'] = np.max(X) * np.random.rand(N, K, L)

	if (type(params['H_init']) != np.ndarray):
		params['H_init'] = (np.max(X) / np.sqrt(T / 3.)) * np.random.rand(K, T)
	else:
		params['H_init'] = zero_pad(params['H_init'], L)  # Pad data


	return X, N, T, K, L, params


def zero_pad(A, L):
	"""
	Adds L columns of zeros to each side of the matrix A.
	"""
	N, M = A.shape
	return np.block([np.zeros([N, L]), A, np.zeros([N,L])])