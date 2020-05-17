
import torch
import torch.nn as nn
from  torch.utils.data import Dataset, DataLoader
import numpy as np

def identity(x):
    return x

def noisy_sin(x, std=0.05):
    return np.sin(x, dtype=np.float64) + np.random.normal(0,std)

def linear(x, scale=0.1):
    return x * scale

class NumericDataset(Dataset):
    """Dataset for generic numerical functions."""

    def __init__(self, samples, fn=identity, data_range=None):
        self.data_range = data_range if data_range is not None else range(0,samples,1)
        self.samples = [(i/samples, fn(i/samples)) for i in self.data_range]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _whiten():
        """
        ZCA Whitening
        """
        tmp_np = np.array(tmp.samples)[:,1]
        cov = np.cov(tmp_np)
        W = cov ** (-1/2)

class SinDataset(NumericDataset):
    def __init__(self, samples, std=0.05, cycles=1.):
        self.data_range = np.linspace(0,2*np.pi, samples)
        self.samples = [(i, noisy_sin(i, std=std)) for i in self.data_range]
