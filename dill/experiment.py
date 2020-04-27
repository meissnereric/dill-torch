from .simple_data import SinDataset
from .simple_model import create_model, init_normal_model
from .training import Trainer
from .utils import get_parameters
from .viz import visualize_predictions, make_predictions
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

gdrive_base_path = '/content/'


def run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                   lr_str="1e4", weight_decay=0, net_width=15, sigma=0.2,
                   rbfs=2, rbf_init='normal',
                   num_epochs=1000, plot=True, gdrive=True, weight_variance=0.01,
                   seed=42, folder_name='exp_data/', record_rate=10_000):
    """
    Experimental code to test for double dip phenomenon.
    Batch size is always the full dataset so SGD == GD.

    """
    import os
    os.makedirs(folder_name, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if gdrive:
        from google.colab import files


    train_dataset = SinDataset(train_samples)
    test_dataset = SinDataset(test_samples, variance=0.0)
    train_loader = DataLoader(train_dataset, batch_size=train_samples)
    test_loader = DataLoader(test_dataset, batch_size=test_samples)
    dataloaders = {'train': train_loader, 'val': test_loader}

    net = create_model(net_width, sigma=sigma, rbfs=rbfs)
    init_normal_model(net, rbfs=rbfs, rbf_init=rbf_init)

    original_basis_params = get_parameters(net, zero_grad=True, param_name='basis')
    original_rbf0_params = get_parameters(net, zero_grad=True, param_name='rbf0')
    original_weight_params = get_parameters(net, zero_grad=False, param_name='weights')

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # full batch size == GD
    criterion = nn.MSELoss()

    trainer = Trainer(dataloaders=dataloaders, model=net, criterion=criterion, optimizer=optimizer)

    trainer.train(num_epochs, verbose=False)

    # Plot and save losses
    train_loss = np.array(trainer.train_loss)
    val_loss = np.array(trainer.val_loss)
    weights_norms = np.array(list(map(lambda x: x.detach().numpy(), trainer.weights_norms)))
    train_loss_file = 'train_loss_lr{}_netwidth{}_weight_variance{}'.format(lr_str, net_width, weight_variance)
    val_loss_file = 'val_loss_lr{}_netwidth{}_weight_variance{}'.format(lr_str, net_width, weight_variance)
    weights_norms_file = 'weights_norms_lr{}_netwidth{}_weight_variance{}'.format(lr_str, net_width, weight_variance)
    np.save(folder_name + train_loss_file, train_loss)
    np.save(folder_name + val_loss_file, val_loss)
    np.save(folder_name + weights_norms_file, weights_norms)
    if gdrive:
        files.download(gdrive_base_path+train_loss_file+'.npy')
        files.download(gdrive_base_path+val_loss_file+'.npy')

    if plot:
        losses_fig_file = 'losses_lr{}_netwidth{}_weight_variance{}.png'.format(lr_str, net_width, weight_variance)
        preds_fig_file = 'preds_lr{}_netwidth{}_weight_variance{}.png'.format(lr_str, net_width, weight_variance)
        pltlen = -1
        start=0
        plt.plot(train_loss[start:pltlen,0], [0] * len(train_loss[start:pltlen,0]), label='perfect loss')
        plt.plot(train_loss[start:pltlen,0], train_loss[start:pltlen,1], label='train loss')
        plt.plot(val_loss[start:pltlen,0], val_loss[start:pltlen,1], label='val loss')
        plt.legend()
        plt.savefig(folder_name + losses_fig_file)
        plt.clf()

        pred, fig = visualize_predictions(net, dataloaders)
        fig.savefig(folder_name + preds_fig_file)
        plt.clf()

        if gdrive:
            files.download(gdrive_base_path+losses_fig_file)
            files.download(gdrive_base_path+preds_fig_file)

    else:
        pred = make_predictions(net, dataloaders)

    return trainer, pred, train_dataset, test_dataset
