import torch
import torch.nn as nn
from .rbf import RBF, gaussian
from .utils import constant_init, apply_init, normal_init, range_init
import numpy as np


def create_rbf_model(net_width, sigma=0.2, hidden_layers=1, linear_hidden=True):
    net = nn.Sequential()
    net.add_module('basis', nn.Linear(1, net_width, bias=False))
    for i in range(hidden_layers):
        net.add_module('rbf{}'.format(i), RBF(net_width, net_width, gaussian, sigma=sigma))
        if linear_hidden and i < (hidden_layers-1):
            net.add_module('linear{}'.format(i), nn.Linear(net_width, net_width, bias=False))

    net.add_module('weights', nn.Linear(net_width, 1, bias=False))
    return net

def create_relu_model(net_width, hidden_layers=1, relu_type='softplus', linear_hidden=True):
    if relu_type=='relu':
        relu = nn.ReLU
    elif relu_type=='softplus':
        relu = nn.Softplus

    net = nn.Sequential()
    net.add_module('basis', nn.Linear(1, net_width, bias=True))
    for i in range(hidden_layers):
        net.add_module('relu{}'.format(i), relu())
        if linear_hidden and i < (hidden_layers-1):
            net.add_module('linear{}'.format(i), nn.Linear(net_width, net_width, bias=False))
    net.add_module('weights', nn.Linear(net_width, 1, bias=False))
    return net

def init_relu_model(model, weight_variance=0.001, basis_variance=0.01, hidden_layers=1, linear_hidden=True):
    apply_init(model, 'basis', constant_init(constant=1))
    apply_init(model, 'weights', normal_init(variance=weight_variance))
    if linear_hidden:
        for i in range(hidden_layers - 1):
            apply_init(model, 'linear{}'.format(i), normal_init(variance=weight_variance))

def init_normal_rbf_model(model, weight_variance=0.01, centres_range=None, hidden_layers=1, init_type='normal', linear_hidden=True):
    centres_range = (0,2*np.pi) if centres_range is None else centres_range
    apply_init(model, 'basis', constant_init(constant=1))
    apply_init(model, 'rbf0', range_init(start=centres_range[0], end=centres_range[1]))
    for i in range(hidden_layers - 1):
        if linear_hidden:
            apply_init(model, 'linear{}'.format(i), normal_init(variance=weight_variance))
        if init_type == 'normal':
            apply_init(model, 'rbf{}'.format(i+1), normal_init(variance=weight_variance))
        elif init_type == 'range':
            apply_init(model, 'rbf{}'.format(i+1), range_init(start=centres_range[0], end=centres_range[1]))
        else:
            apply_init(model, 'rbf{}'.format(i+1), constant_init(constant=1))

    apply_init(model, 'weights', normal_init(variance=weight_variance))
