import torch
import torch.nn as nn
import numpy as np
from .viz import plot_outputs
from .simple_model import create_model, init_const_model
import torch.nn.init as init
from .rbf import RBF, gaussian
from .utils import constant_init, apply_init

# What is the output at layer 1
def test_layer_one_outputs(data, net_width=5):
    one_layer = nn.Linear(1, net_width, bias=False)
    init.constant_(one_layer.weight.data, 1)
    output = one_layer(data)
    plot_outputs(output)

# What is the output at layer 2
def test_layer_two_outputs(data, net_width=5, sigma=0.2):
    two_layer = nn.Sequential()
    two_layer.add_module('basis', nn.Linear(1, net_width, bias=False))
    # net.add_module('relu', nn.ReLU())
    two_layer.add_module('rbf', RBF(net_width, net_width, gaussian, sigma=sigma))
    apply_init(two_layer, 'basis', constant_init(constant=1))
    output = two_layer(data)
    plot_outputs(output)


# What is the final output at layer 3
def test_full_simple_model(data, net_width=5, sigma=0.2):
    net = create_model(net_width, sigma)
    init_const_model(net)
    outputs = net(data)
    plot_outputs(outputs, final_layer=True)
