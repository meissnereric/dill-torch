
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import torch.nn as nn

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

def plot_dataset(train_dataset, test_dataset, ax):
    test_np = np.array(test_dataset.samples)
    ax.plot(test_np[:,0], test_np[:,1], label='true function',
            color='b')
    train_np = np.array(train_dataset.samples)
    ax.scatter(train_np[:,0], train_np[:,1],
            label='training points', marker='x', color='r')
    return ax

def visualize_predictions(net, train_dataset, test_dataset, dataloaders):
    predictions = make_predictions(net, dataloaders)
    ax = plt.axes()
    plot_dataset(train_dataset, test_dataset, ax)
    ax.plot(predictions[0][0], predictions[0][2], label='predictions',
            color='g')
    ax.legend()

    return predictions, plt.gcf()

def plot_basis(net, test_tensor, ax, title=None):
    output = net(test_tensor)
    for sample in output.T:
        sample = sample.detach().numpy()
        ax.plot(test_tensor, sample)
    ax.set_title(title)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are

    # ax.set_xticks(range(0, len(test_tensor), int(len(test_tensor) / 8.)))
    # tt = test_tensor.numpy()[0::int(len(test_tensor)/8)]
    # ax.set_xticklabels(list(map(lambda x: "{:02.2f}".format(x[0]), tt)))

    return output, ax

def plot_network_outputs_across_layers(seq_net, test_tensor,
                train_dataset, test_dataset, dataloaders,
                save=True, folder_name='exp_data/', file_name='network_outputs_'):
    fig = plt.figure(figsize=(7, 20))
    gspec = fig.add_gridspec(ncols=1, nrows=5)
    tmp_mods = OrderedDict()
    for i, (name, mod) in enumerate(seq_net.named_modules()):
        if len(name) > 0:
            ax = fig.add_subplot(gspec[i-1, 0])
            tmp_mods[name] = mod
            output, ax = plot_basis(nn.Sequential(tmp_mods), test_tensor, ax, title='Outputs at layer {}'.format(name))

    plot_dataset(train_dataset, test_dataset, ax)

    ax.legend(loc=(1,1))
    if save:
        fig.savefig(folder_name +  file_name + 'layer{}.png'.format(name))
    fig.show()
