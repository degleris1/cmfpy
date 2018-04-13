import numpy as np
from seqnmf import seq_nmf
import matplotlib.pyplot as plt
from munkres import Munkres


def seq_nmf_data(N, T, L, K):
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
    W[W < .5] = 0
    H[H < .8] = 0
    lrd = np.dot(W, H)

    # add a random shift to each row
    lags = np.random.randint(0, L, size=N)
    data = np.array([np.roll(row, l, axis=-1) for row, l in zip(lrd, lags)])
    # data = lrd

    return data, W, H


def test_seq_nmf():
    data, realW, realH = seq_nmf_data(100, 101, 10, 5)
    realH /= np.linalg.norm(realH, axis=-1, keepdims=True)

    losses = []

    for k in range(1, 10):
        if k == 5:
            W, H, costhist, loadings, power = seq_nmf(data, K=k, L=20, H_init=realH)
        else:
            W, H, costhist, loadings, power = seq_nmf(data, K=k, L=20)

        losses.append(power)

        if k == 5:
            estH = H
            estW = W

    # Use Munkres algorithm to match rows of H and estH
    matchcost = 1 - np.dot(realH, estH.T)
    indices = Munkres().compute(matchcost.copy())
    _, prm_est = zip(*indices)
    estH = estH[list(prm_est)]

    print('Hdiff: ', np.linalg.norm(estH - realH) / np.linalg.norm(realH))

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(realH / np.linalg.norm(realH, axis=-1, keepdims=True))
    axes[1].imshow(estH)

    plt.figure()
    plt.imshow(data, cmap='gray')

    plt.figure()
    plt.plot(np.arange(len(losses))+1, losses)
    plt.xlabel('rank')
    plt.ylabel('cost')
    plt.show()
