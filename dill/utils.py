import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from .rbf import RBF

def normal_init(mean=0., std=0.01):
    """
    std will be scaled by the dimensionality of the layer.
    """
    def norm_init(m):
        if isinstance(m, nn.Linear):
            dim = m.weight.data.size()
            init.normal_(m.weight.data, mean=mean, std=(std / dim[1]))
        if isinstance(m, RBF):
            dim = m.centres.data.size()
            init.normal_(m.centres.data, mean=mean, std=(std / dim[1]))
    return norm_init

def constant_init(constant=1):
    def const_init(m):
        if isinstance(m, nn.Linear):
            init.constant_(m.weight.data, constant)
            if m.bias is not None:
                trange = torch.Tensor(np.linspace(-2*np.pi, 0, num=m.bias.data.size()[0]))
                trange = trange.reshape(m.bias.data.size())
                m.bias.data = trange
        if isinstance(m, RBF):
            init.constant_(m.centres.data, constant)

    return const_init

def range_init(start=-3, end=3):
    def rng_init(m, start=start, end=end):
        if isinstance(m, nn.Linear):
            trange = torch.Tensor(np.linspace(start, end, num=m.weight.data.size()[0]))
            trange = trange.reshape(m.weight.data.size())
            m.weight.data = trange
        if isinstance(m, RBF):
            size=m.centres.data.size()[1]
            trange = torch.Tensor(np.linspace(start, end, num=size))
            trange = trange.reshape(size)
            m.centres.data = trange
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

def get_parameters(net, zero_grad=False, layer_name='basis', param_name=None):
    """
    :param zero_grad: If True, will zero the gradients of all parameters in the requested layer.
    :param param_name: If param_name is None, will turn off the grads of all parameters in that layer.
    """
    for mname, mod in net.named_modules():
        if mname == layer_name:
            params = [i for i in mod.named_parameters()]
            if zero_grad:
                mod.requires_grad = False
                for pname, param in params:
                    if param_name is None or pname == param_name:
                        param.requires_grad = False
            return params
