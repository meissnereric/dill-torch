import torch
import torch.nn as nn
import numpy as np
from .viz import plot_outputs
from .simple_model import create_model, init_const_model

# What is the output at layer 1
def test_layer_one_outputs(data):
    one_layer = nn.Linear(1, net_width, bias=False)
    init.constant_(one_layer.weight.data, 1)
    output = one_layer(data)
    plot_outputs(output)

# What is the output at layer 2
def test_layer_two_outputs(data):
    two_layer = nn.Sequential()
    two_layer.add_module('basis', nn.Linear(1, net_width, bias=False))
    # net.add_module('relu', nn.ReLU())
    two_layer.add_module('rbf', RBF(net_width,net_width,gaussian))
    apply_init(two_layer, 'basis', constant_init)
    output = two_layer(data)
    plot_outputs(output)


# What is the final output at layer 3
def test_full_simple_model(data):
    net_width = 5
    net = create_model(net_width)
    init_const_model(net)
    outputs = net(data)
    plot_outputs(outputs, final_layer=True)

def visualize_predictions(net, dataloaders):
    predictions = []
    for (inp, label) in dataloaders['val']:
        inputs = inp.reshape(inp.shape[0],1)
        pred = net(inputs).reshape(inp.shape[0])
        predictions.append([inp.numpy(), label.numpy(), pred.detach().numpy()])
    predictions = np.array(predictions)

    print("Predictions: {}".format(predictions))

    ax = plt.axes()
    ax.plot(predictions[0][0], predictions[0][1], label='true function')
    ax.plot(predictions[0][0], predictions[0][2], label='predictions')
    ax.legend()

    return predictions
