import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

def normal_init(m):
    if isinstance(m, nn.Linear):
        normal_(m.weight.data, mean=0.0, std=0.1)
        # normal_(m.bias.data, mean=0.0, std=0.1)

def constant_init(m, constant=1):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight.data, constant)
        # init.constant_(m.bias.data, 1)

def range_init(m, start=-3, end=3):
    if isinstance(m, nn.Linear):
        trange = torch.Tensor(np.linspace(start, end, num=m.weight.data.size()[0]))
        trange = trange.reshape(m.weight.data.size())
        m.weight.data = trange
        # init.constant_(m.bias.data, 0)

def apply_init(net, name, fn):
    for n, mod in net.named_modules():
        if n == name:
            print(mod.weight.size())
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
            print("mod params")
            if zero_grad:
                print('Setting {} gradients to 0'.format(param_name))
                mod.requires_grad = False
                for param in params:
                    param.requires_grad = False
            # print('basis parameters {}'.format(basis_grads))
            return params
