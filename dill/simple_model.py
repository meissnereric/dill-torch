import torch
import torch.nn as nn
from .rbf import RBF, gaussian
from .utils import constant_init, apply_init
from .viz import plot_outputs


def create_model(net_width):
    net = nn.Sequential()
    net.add_module('basis', nn.Linear(1, net_width, bias=False))
    net.add_module('rbf', RBF(net_width, net_width, gaussian))
    net.add_module('weights', nn.Linear(net_width, 1, bias=False))
    return net

def init_const_model(model):
    apply_init(model, 'basis', constant_init)
    apply_init(model, 'weights', constant_init)
