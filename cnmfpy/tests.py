import numpy as np
import numpy.linalg as la

from cnmfpy.cnmf import CNMF
import cnmfpy.regularize as reg
from cnmfpy.conv import ShiftMatrix

import matplotlib.pyplot as plt
#from munkres import Munkres


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
    # data = lrd

    return data, W, H


def test_seq_nmf(N=100, T=120, L=10, K=5):
    data, realW, realH = seq_nmf_data(N, T, L, K)
    realH /= la.norm(realH, axis=-1, keepdims=True)

    losses = []
    for k in range(1, 2*K+1):
        if (k == K):
            W, H, costhist, loadings, power = seq_nmf(data, K=k, L=2*L, lam=10**(-6), maxiter=200, H_init=realH)
        else:
            W, H, costhist, loadings, power = seq_nmf(data, K=k, L=2*L, lam=10**(-6), maxiter=200)

        losses.append(power)

        if (k == K):
            estH = H
            estW = W

    # Use Munkres algorithm to match rows of H and estH
    #matchcost = 1 - np.dot(realH, estH.T)
    #indices = Munkres().compute(matchcost.copy())
    #_, prm_est = zip(*indices)
    #estH = estH[list(prm_est)]

    #print('Hdiff: ', np.linalg.norm(estH - realH) / np.linalg.norm(realH))
    error = data - _reconstruct(estW, estH)
    print('Percent error: ', la.norm(error)**2 / la.norm(data)**2)

    # Plot real H vs estimated H
    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(realH / la.norm(realH, axis=-1, keepdims=True))
    axes[1].imshow(estH)

    # Plot reconstruction error
    plt.figure()
    plt.imshow(np.abs(error), cmap='gray')
    plt.colorbar()

    # Plot losses
    plt.figure()
    plt.plot(np.arange(len(losses))+1, losses)
    plt.xlabel('rank')
    plt.ylabel('cost')
    plt.show()



if (__name__ == '__main__'):
    data, W, H = seq_nmf_data(100, 300, 10, 2)

    losses = []

    K = 2
    for k in range(1, K+1):
        model = CNMF(k, 10).fit(data, alg='bcd')
        plt.plot(model.loss_hist[1:])
        losses.append(model.loss_hist[-1])

    plt.figure()
    plt.plot(range(1,K+1), losses)

    plt.figure()
    plt.imshow(model.predict())
    plt.title('Predicted')

    plt.figure()
    plt.imshow(data)

    plt.show()