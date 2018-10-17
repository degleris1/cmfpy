import numpy as np  # Linear algebra


def init_rand(model, data):
    """
    Random initialization.
    """

    n_neurons, n_time = data.shape

    mag = np.amax(data)

    W = mag * np.abs(np.random.rand(model.maxlag, n_neurons,
                                    model.n_components))
    H = mag * np.abs(np.random.rand(model.n_components, n_time))

    return W, H
