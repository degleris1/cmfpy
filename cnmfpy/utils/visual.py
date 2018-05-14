import numpy as np
import matplotlib.pyplot as plt

EPSILON = np.finfo(np.float32).eps


def sort_neurons(W):
    """
    Sort neurons, first by factor, then by weight.
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
    Plot the components of H.
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
