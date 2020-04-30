from dill.experiment import run_experiment
from dill.simple_model import create_relu_model
from dill.utils import *
import numpy as np
import shutil
data_folder='tst/test_outputs'


epochs=100 * 1_000
net_width = 30
hidden_layers = 1
relu_type='softplus'
weight_variance=0.001
basis_variance=0.01

net = create_relu_model(net_width, hidden_layers=hidden_layers, relu_type=relu_type)

apply_init(net, 'basis', normal_init(variance=basis_variance))
apply_init(net, 'weights', normal_init(variance=weight_variance))

trainer1, pred, train1, test1 = run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                   lr_str="1e4", weight_decay=0, net_width=net_width, sigma=0.2,
                   num_epochs=epochs, plot=False, gdrive=False, weight_variance=0.01,
                   folder_name=data_folder+'/', hidden_layers=hidden_layers, layer_type='relu',
                   relu_type=relu_type, net=net,
                   print_rate=1_000)

shutil.rmtree(data_folder, ignore_errors=True)
