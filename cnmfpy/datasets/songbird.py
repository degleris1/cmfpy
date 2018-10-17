import numpy as np
from scipy.io import loadmat
import os

DATAPATH = os.path.expanduser("~/cmf_data/MackeviciusData.mat")


class SongbirdHVC:
    """Calcium imaging data from neurons in Higher Vocal Centre (HVC)."""

    def __init__(self, normalize=True, path=DATAPATH):
        self.name = "songbird_hvc"
        self.normalize = normalize
        self.path = path

    def generate(self):
        data = loadmat(self.path)["NEURAL"]
        if self.normalize:
            data /= 1e-6 + np.linalg.norm(data, axis=1, keepdims=True)
        return data
