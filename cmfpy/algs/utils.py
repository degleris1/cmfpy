"""
Utilities for optimization algorithms.
"""
import numba


@numba.jit(nopython=True, cache=True)
def param_convergence(x, y):
    """
    Computes convergence on parameters.

    Parameters
    ----------
    x : ndarray
        1d array of parameters, current estimate.
    y : ndarray
        1d array of parameters, last estimate.
    """

    min_df = x[0] - y[0]
    max_df = min_df
    min_x = x[0]
    max_x = x[0]

    for xi, yi in zip(x[1:], y[1:]):
        df = xi - yi
        if df < min_df:
            min_df = df
        if df > max_df:
            max_df = df
        if xi < min_x:
            min_x = xi
        if xi > max_x:
            max_x = xi

    return (max_df - min_df) / (max_x - min_x)
