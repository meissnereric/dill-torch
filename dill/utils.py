import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

def normal_init(mean=0., noise=0.01):
    """
    Noise will be scaled by the dimensionality of the layer.
    """
    def norm_init(m):
        if isinstance(m, nn.Linear):
            dim = m.weight.data.size()
            init.normal_(m.weight.data, mean=mean, std=(noise / dim[1]))
    return norm_init

def constant_init(constant=1):
    def const_init(m):
        if isinstance(m, nn.Linear):
            init.constant_(m.weight.data, constant)
    return const_init
        # init.constant_(m.bias.data, 1)

def range_init(start=-3, end=3):
    def rng_init(m, start=-3, end=3):
        if isinstance(m, nn.Linear):
            trange = torch.Tensor(np.linspace(start, end, num=m.weight.data.size()[0]))
            trange = trange.reshape(m.weight.data.size())
            m.weight.data = trange
    return rng_init

def apply_init(net, name, fn, **kwargs):
    for n, mod in net.named_modules():
        if n == name:
            mod.apply(fn)

def compute_layer_norm(net, layer_name='weights', norm=2):
    """
    Takes the norm of each parameters in a module
    separately and returns a list of the norms.
    """
    norms = []
    for name, mod in net.named_modules():
        if name == layer_name:
            for i in mod.parameters():
                norms.append(i.norm(norm))
            break
    return norms

def get_parameters(net, zero_grad=False, param_name='basis'):
    for name, mod in net.named_modules():
        if name == param_name:
            params = [i for i in mod.parameters()]
            if zero_grad:
                mod.requires_grad = False
                for param in params:
                    param.requires_grad = False
            return params
