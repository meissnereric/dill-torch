import torch
import torch.nn as nn
from .rbf import RBF, gaussian
from .utils import constant_init, apply_init, normal_init, range_init
from .viz import plot_outputs
import numpy as np


def create_model(net_width, sigma=0.2, rbfs=1):
    net = nn.Sequential()
    net.add_module('basis', nn.Linear(1, net_width, bias=False))
    for i in range(rbfs):
        net.add_module('rbf{}'.format(i), RBF(net_width, net_width, gaussian, sigma=sigma))
    net.add_module('weights', nn.Linear(net_width, 1, bias=False))
    return net

def init_const_model(model, centres_range=None, rbfs=1):
    centres_range = (0,2*np.pi) if centres_range is None else centres_range
    apply_init(model, 'basis', constant_init(constant=1))
    apply_init(model, 'rbf0', range_init(start=centres_range[0], end=centres_range[1]))
    for i in range(rbfs - 1):
        apply_init(model, 'rbf{}'.format(i+1), constant_init(constant=1))
    apply_init(model, 'weights', constant_init(constant=1))

def init_normal_model(model, weight_variance=0.01, centres_range=None, rbfs=1, rbf_init='normal'):
    centres_range = (0,2*np.pi) if centres_range is None else centres_range
    apply_init(model, 'basis', constant_init(constant=1))
    apply_init(model, 'rbf0', range_init(start=centres_range[0], end=centres_range[1]))
    for i in range(rbfs - 1):
        if rbf_init == 'normal':
            apply_init(model, 'rbf{}'.format(i+1), normal_init(variance=weight_variance))
        elif rbf_init == 'range':
            apply_init(model, 'rbf{}'.format(i+1), range_init(start=centres_range[0], end=centres_range[1]))
        else:
            apply_init(model, 'rbf{}'.format(i+1), constant_init(constant=1))
    apply_init(model, 'weights', normal_init(variance=weight_variance))
