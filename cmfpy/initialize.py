import numpy as np  # Linear algebra

from .conv import tensor_conv


def init_rand(model, data):
    """
    Random initialization with appropriate scaling
    """

    n_neurons, n_time = data.shape

    W = np.abs(np.random.rand(model.maxlag, n_neurons, model.n_components))
    H = np.abs(np.random.rand(model.n_components, n_time))

    est = tensor_conv(W, H)

    # TODO: epsilon necessary?
    alpha = np.dot(data.ravel(), est.ravel()) / np.linalg.norm(est)**2

    return alpha * W, alpha * H
