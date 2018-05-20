import numpy as np
import matplotlib.pyplot as plt


import cnmfpy.regularize as reg
from cnmfpy.cnmf import CNMF
from cnmfpy.conv import ShiftMatrix
from cnmfpy.utils.visual import sort_neurons



def seq_nmf_data(N, T, L, K, sparsity=0.8):
    """Creates synthetic dataset for conv NMF

    Args
    ----
    N : number of neurons
    T : number of timepoints
    L : max sequence length
    K : number of factors / rank

    Returns
    -------
    data : N x T matrix
    """

    # low-rank data
    W, H = np.random.rand(N, K), np.random.rand(K, T)
    W[W < sparsity] = 0
    H[H < sparsity] = 0
    lrd = np.dot(W, H)

    # add a random shift to each row
    lags = np.random.randint(0, L, size=N)
    data = np.array([np.roll(row, l, axis=-1) for row, l in zip(lrd, lags)])

    return data, W, H


if (__name__ == '__main__'):
    data, W, H = seq_nmf_data(100, 300, 10, 2)

    losses = []

    K = 2
    for k in range(1, K+1):
        model = CNMF(k, 10, tol=0, n_iter_max=1000).fit(data, alg='mult')
        plt.plot(model.loss_hist[1:])
        losses.append(model.loss_hist[-1])

    plt.figure()
    plt.plot(range(1,K+1), losses)

    ordering = sort_neurons(model.W)

    plt.figure()
    plt.imshow(model.predict()[ordering,:])
    plt.title('Predicted')

    plt.figure()
    plt.imshow(data[ordering,:])

    plt.show()