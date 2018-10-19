import numpy as np
import time
from tqdm import trange

from ..optimize import compute_loss, renormalize

from .mult import mult_step
from .chals import chals_step


ALGORITHMS = {'mult': mult_step, 'chals': chals_step}


def fit_alg(data, model, update_rule):
    m, n = data.shape

    # initial loss
    model.loss_hist = [compute_loss(data, model.W, model.H)]
    model.time_hist = [0.0]
    t0 = time.time()

    itr = 0
    for itr in trange(model.n_iter_max):

        if (np.isnan(model.W).any()):
            raise Exception('W has Nans!!')

        if (np.isnan(model.H).any()):
            raise Exception('H has NANs!!')

        update_rule(data, model)

        model.loss_hist.append(compute_loss(data, model.W, model.H))
        model.time_hist.append(time.time() - t0)

        # renormalize H
        model.W, model.H = renormalize(model.W, model.H)

        # check convergence
        prev_loss, new_loss = model.loss_hist[-2:]
        if (np.abs(prev_loss - new_loss) < model.tol):
            break
