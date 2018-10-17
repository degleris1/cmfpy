"""Specifies dimensions of the model and data."""


class ModelDimensions:
    """Holds dimensions of CMF model."""

    def __init__(self, data=None, n_features=None, n_timepoints=None,
                 n_lags=None, n_components=None):
        """
        Parameters
        ----------
            data : ndarray
                Matrix holding time series, shape: (n_features, n_timepoints).
                If not specified, must provide both `n_features` and
                `n_timepoints`.
            n_features : int
                Number of features/measurements recorded in time series.
                Must be specified if `data` is not given.
            n_timepoints : int
                Length of time series. Must be specified if `data` is not
                given.
            n_lags : int
                Number of time steps in each temporal motifs/sequence. Must be
                specified.
            n_components : int
                Number of temporal motifs/sequences. Must be specified.
        """

        if data is None:
            if (n_features is None) or (n_timepoints is None):
                raise ValueError("Must either specify 'data' or ('n_features' "
                                 "and 'n_timepoints').")
        else:
            n_features, n_timepoints = data.shape

        if n_lags is None:
            raise ValueError("Must specify 'n_lags'.")

        if n_components is None:
            raise ValueError("Must specify 'n_components'.")

        self.n_features = n_features
        self.n_timepoints = n_timepoints
        self.n_lags = n_lags
        self.n_components = n_components
