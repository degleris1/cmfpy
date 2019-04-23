"""
Initialization procedures for CMF.
"""

import numpy as np
import numpy.random as npr
from .common import tconv


def rand(data, dims, rs):
    """
    Random initialization with appropriate scaling.
    """

    # Generate random nonnegative parameters.
    W = rs.rand(dims.maxlag, dims.n_units, dims.n_components)
    H = rs.rand(dims.n_components, dims.n_timepoints)

    # Appropriately rescale initialization
    est = tconv(W, H)
    alpha = (np.linalg.norm(data) / np.linalg.norm(est)) ** .5
    W *= alpha
    H *= alpha

    return W, H
