
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import torch.nn as nn

def plot_outputs(outputs, final_layer=False, verbose=False):
    """
    Accepts a torch.Tensor and plots it.
    """
    if verbose:
        print('***************************************')
        print(outputs)
    if not final_layer:
        for sample in outputs.T:
            sample = sample.detach().numpy()
            plt.plot(sample)
    else:
        plt.plot(outputs.detach().numpy())
    if verbose:
        print('***************************************')

def make_predictions(net, dataloaders, verbose=False):
    predictions = []
    for (inp, label) in dataloaders['val']:
        inputs = inp.reshape(inp.shape[0],1)
        pred = net(inputs).reshape(inp.shape[0])
        predictions.append([inp.numpy(), label.numpy(), pred.detach().numpy()])
    predictions = np.array(predictions)

    if verbose:
        print("Predictions: {}".format(predictions))
    return predictions

def visualize_predictions(net, dataloaders):
    predictions = make_predictions(net, dataloaders)
    ax = plt.axes()
    ax.plot(predictions[0][0], predictions[0][1], label='true function')
    ax.plot(predictions[0][0], predictions[0][2], label='predictions')
    ax.legend()

    return predictions, plt.gcf()

def plot_basis(net, test_tensor, title=None):
    output = net(test_tensor)
    plot_outputs(output)
    plt.title(title)
    plt.xticks(range(0, 200, 25), list(map(lambda x: "{:02.2f}".format(x), np.linspace(-1, 7, 200))))
    plt.show()

def plot_individual_output_layers(seq_net, test_tensor, save=True, folder_name='exp_data/', file_name='network_outputs_'):
    tmp_mods = OrderedDict()
    for i, (name, mod) in enumerate(seq_net.named_modules()):
        if len(name) > 0:
            tmp_mods[name] = mod
            plot_basis(nn.Sequential(tmp_mods), test_tensor, title='Outputs at layer {}'.format(name))
            if save:
                plt.savefig(folder_name +  file_name + 'layer{}.png'.format(name))
