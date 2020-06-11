import matplotlib.image as mpimg
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

class ExpResults:
    def __init__(self, network_type, basepath='/content/'):
        self.basepath = basepath
        self.network_type = network_type
        self.parameters = self.load_parameters(basepath + network_type + '_parameters.pkl')

        self.train_losses = np.load(basepath + 'train_agg.npy')
        self.val_losses = np.load(basepath + 'val_agg.npy')
        self.weight_norms = np.load(basepath + 'norms_agg.npy')

    def load_parameters(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)



def plot_basis_image(base_path, lr, weight_std, net_width, figsize=(9,25)):
    """
    Plots the basis plot image passed in as a mpimg in matplotlib.
    """
    plt.figure(figsize=figsize)
    img=mpimg.imread(base_path.format(lr, weight_std, net_width))
    imgplot = plt.imshow(img)
    plt.title("LR {} StdDev {} Width {}".format(lr, weight_std, net_width))

def plot_losses(losses,
            nw_index, nw,
            lr_index, lr,
            std_index, std,
            start=0, end=-1,
            legend_string="NW {} LR {} Std {}",
            ylim=None
            ):
    """
    Plots a single training or validation loss trajectory
    for the indexes given.
    """
    xaxis = losses[nw_index, lr_index, std_index,start:end,0]
    yaxis = losses[nw_index, lr_index, std_index,start:end,1]
    plt.plot(xaxis, yaxis, label=legend_string.format(nw, lr, std))

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if ylim =='log':
        plt.yscale('log')
    elif ylim is not None:
        plt.ylim(ylim)

    plt.legend(loc=(1,0))

def plot_best(losses,
            lr_index, lr,
            std_index, std,
            loss_type,
            start=0, end=-1,
            legend_string="Best {} Std {} LR {}",
            ylim=None
            ):
    """
    Plots the best loss values across all of training
     for the indexes given, across net widths.
    """
    best_loss = np.min(losses[:,lr_index,std_index,:,1], axis=1)
    plt.plot(best_loss, label=legend_string.format(loss_type, std, lr))

def plot_final(losses,
            lr_index, lr,
            std_index, std,
            loss_type,
            start=0, end=-1,
            legend_string="Final {} Std {} LR {}",
            ylim=None
            ):
    """
    Plots the final loss values for the indexes given across net widths.
    """
    final_loss = losses[:,lr_index,std_index,-1,1]
    plt.plot(final_loss, label=legend_string.format(loss_type, std, lr))
