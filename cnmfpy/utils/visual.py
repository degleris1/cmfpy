import numpy as np
import matplotlib.pyplot as plt

EPSILON = np.finfo(np.float32).eps


def sort_neurons(W):
    """
    Sorts neurons, first by factor, then by time lag within factor.
    Timelags are defined as the center of mass within a particular factor.

    Parameters
    ----------
    W : array-like, shape (n_lag, n_neurons, n_components)
        Tensor of neural sequences produced by the CNMF model.

    Returns
    -------
    ordering : list, ints
        Indices of the neurons sorted by factor, then by timelag within factor.
    """
    L, N, K = W.shape

    # compute the energy of each neuron in each factor
    energies = np.sum(W, axis=0)
    best_factors = np.argmax(energies, axis=1)

    # sort by factor
    factor_sort = [x for x in sorted(zip(best_factors, range(N)))]

    # compute 'center' of each neuron in each factor
    lags = np.zeros((N, K))
    ind = np.arange(L)

    for k in range(K):
        for n in range(N):
            lags[n, k] = int(ind.dot(W[:, n, k]) / (energies[n, k] + EPSILON))

    neural_sort = sorted(factor_sort, key = lambda x : (x[0], lags[x[1], x[0]]))

    return [x[1] for x in neural_sort]


def plot_H_components(H, n_components=None, figsize=(16,4), color='b'):
    """
    Plots the components of H.

    Parameters
    ----------
    H : array-like, shape (n_components, n_time)
        Matrix of time sequences produced by the CNMF model.
    n_components : int, optional
        Number of components to plot. By default, plot all the components.
    figsize : tuple, (int, int), optional
        Two tuple of ints. Size of the figure to plot.
    color : string, optional
        Color of the components. By default, plots are blue.
    """
    if (n_components == None):
        n_components = H.shape[0]
    elif (n_components > H.shape[0]):
        raise ValueError('Too many components.')

    plt.figure(figsize=figsize)

    for i in range(n_components):
        plt.subplot(n_components, 1, i+1)
        plt.plot(H[i], c=color)

    plt.show()
