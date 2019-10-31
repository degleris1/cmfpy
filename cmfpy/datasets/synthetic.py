import numpy as np
import numpy.random as npr
import scipy.stats
from cmfpy.common import tconv


def synthetic_point_process_data(
    n_components=2, n_units=100, maxtime=10.0, edge_padding=3.0,
    unit_background_rates=0.01, unit_mean_amp=1.0, unit_sparsity=1e-2,
    unit_delay_sigma=0.5, unit_response_width=0.1, tw_log_sigma=1e-5,
    latent_event_rate=1.0, latent_ev_times=None, latent_ev_types=None,
    latent_ev_warps=None, event_amp_log_sigma=1e-1, seed=None):

    rs = npr.RandomState(seed)

    spike_times = []
    spike_units = []

    # Sample parameters
    unit_delays = rs.normal(0.0, unit_delay_sigma, size=(n_units, n_components))
    unit_amps = rs.exponential(unit_mean_amp, size=(n_units, 1)) * \
        rs.dirichlet(np.full(n_components, unit_sparsity), size=n_units)

    unit_cluster = np.argmax(unit_amps, axis=1)
    unit_delay_in_cluster = unit_delays[np.arange(n_units), unit_cluster]
    neuron_perm = np.lexsort(np.row_stack(
        (-unit_delay_in_cluster, unit_cluster)))

    unit_amps = unit_amps[neuron_perm]
    unit_delays = unit_delays[neuron_perm]

    # Sample latent events.
    num_latent_events = rs.poisson(
        n_components * maxtime * latent_event_rate)
    if latent_ev_times is None:
        latent_ev_times = rs.uniform(0, maxtime, size=num_latent_events)
    if latent_ev_types is None:
        latent_ev_types = rs.randint(n_components, size=num_latent_events)
    if latent_ev_warps is None:
        latent_ev_warps = np.exp(rs.normal(0.0, tw_log_sigma, size=num_latent_events))

    # Sample spikes arising from latent events.
    for t, typ, warp in zip(latent_ev_times, latent_ev_types, latent_ev_warps):
        print(t)
        for n in range(n_units):
            normdist = scipy.stats.norm(
                t + warp * unit_delays[n, typ], unit_response_width)
            normdist.random_state = rs
            n_spk = rs.poisson(unit_amps[n, typ])
            spike_times.append(normdist.rvs(size=n_spk))
            spike_units.append(np.full(n_spk, n))


    # Sample background events.
    n_bck_spikes = rs.poisson(
        n_units * unit_background_rates * maxtime + 2 * edge_padding)
    spike_times.append(
        rs.uniform(-edge_padding, edge_padding + maxtime, size=n_bck_spikes))
    spike_units.append(
        rs.randint(n_units, size=n_bck_spikes))

    return np.concatenate(spike_times), np.concatenate(spike_units)


class Synthetic:
    """Synthetic data."""
    def __init__(self,
                 n_components=3,
                 n_units=100,
                 n_lags=100,
                 n_timebins=10000,
                 H_sparsity=0.9,
                 noise_scale=1.0,
                 seed=None):

        # Set data name and random state.
        self.name = "synthetic"
        self.rs = np.random.RandomState(seed)

        # Generate random convolutional parameters
        W = np.zeros((n_lags, n_units, n_components))
        H = self.rs.rand(n_components, n_timebins)

        # Add sparsity to factors
        self.H = H * self.rs.binomial(1, 1 - H_sparsity, size=H.shape)

        # Add structure to motifs
        for i, j in enumerate(np.random.choice(n_components, size=n_units)):
            W[:, i, j] += _gauss_plus_delay(n_lags)
        self.W = W

        # Determine noise
        self.noise = noise_scale * self.rs.rand(n_units, n_timebins)

        # Add noise to model prediction
        self.data = tconv(self.W, self.H) + self.noise

    def generate(self):
        return self.data + self.noise


def _gauss_plus_delay(n_steps):
    tau = np.random.uniform(-1.5, 1.5)
    x = np.linspace(-3-tau, 3-tau, n_steps)
    y = np.exp(-x**2)
    return y / y.max()
