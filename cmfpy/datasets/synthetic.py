import numpy as np


class Synthetic:
    """Synthetic data."""
    def __init__(self,
                 n_components=3,
                 n_features=100,
                 n_lags=10,
                 n_timebins=1000,
                 sparsity=0.8,
                 seed=None):

        self.name = "synthetic"

        self.random_state = np.random.RandomState(seed)

        # Low rank factors
        self.W = self.random_state.rand(n_features, n_components)
        self.H = self.random_state.rand(n_components, n_timebins)
        self.W[self.W < sparsity] = 0
        self.H[self.H < sparsity] = 0

        # Create data by shifting each row by a random offset
        lags = self.random_state.randint(0, n_lags, size=n_features)
        data = []
        for row, lag in zip(np.dot(self.W, self.H), lags):
            data.append(np.roll(row, lag, axis=-1))
        self.data = np.array(data)

    def generate(self):
        return self.data
