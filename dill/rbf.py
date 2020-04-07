# RBF Layer

import torch
import torch.nn as nn
import numpy as np

class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        sigmas: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func, sigma=0.2, verbose=False, centres_range=None):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(1, out_features))
        self.sigmas = torch.Tensor([sigma])
        self.basis_func = basis_func
        self.centres_range = (0,2*np.pi) if centres_range is None else centres_range
        self.reset_parameters(in_features, self.centres_range, sigma)
        self.verbose=verbose

    def reset_parameters(self, in_features, centres_range, sigma=0.2):
        rng = np.linspace(centres_range[0], centres_range[1], in_features)
        self.centres.data = torch.Tensor(rng)
        nn.init.constant_(self.sigmas, sigma)

    def forward(self, input):
        size = (input.size(0), self.out_features)#, self.in_features)
        x = input.expand(size)
        c = self.centres.expand(size)
        a = (x - c).pow(2).pow(0.5)
        b = a / (self.sigmas)
        distances = b
        if self.verbose:
            print('sigma', self.sigmas)
            print('(input) x', x)
            print('(c) centres', c)
            print('size', size)
            print('a', a)
            print('b', b)
            print('output', self.basis_func(distances))
        return self.basis_func(distances)

# RBFs

def gaussian(alpha):
    phi = torch.exp((-1/2)*alpha.pow(2))
    return phi
