from .simple_data import SinDataset
from .simple_model import create_rbf_model, init_normal_rbf_model, create_relu_model, init_relu_model
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
                   weight_decay=0, net_width=15, sigma=0.2,
                   hidden_layers=2, init_type='normal',
                   num_epochs=1000, plot=True, gdrive=True, weight_std=0.001,
                   seed=42, folder_name='exp_data/', record_rate=1000, print_rate=10000,
                   layer_type='rbf', relu_type='softplus', basis_std=0.01,
                   net=None, linear_hidden=True,
                   dataloaders=None, fix_basis=True):
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

    lr_str =  str(learning_rate)

    if dataloaders is None:
        train_dataset = SinDataset(train_samples)
        test_dataset = SinDataset(test_samples, std=0.0)
        train_loader = DataLoader(train_dataset, batch_size=train_samples)
        test_loader = DataLoader(test_dataset, batch_size=test_samples)
        dataloaders = {'train': train_loader, 'val': test_loader}
    else:
        train_dataset = None
        test_dataset = None

    if net is None:
        if layer_type=='rbf':
            net = create_rbf_model(net_width, sigma=sigma, hidden_layers=hidden_layers, linear_hidden=linear_hidden)
            init_normal_rbf_model(net, hidden_layers=hidden_layers, init_type=init_type, weight_std=weight_std, linear_hidden=linear_hidden)

        else:
            net = create_relu_model(net_width, hidden_layers=hidden_layers, relu_type=relu_type, linear_hidden=linear_hidden)
            init_relu_model(net, weight_std=weight_std, basis_std=basis_std, hidden_layers=hidden_layers, linear_hidden=linear_hidden)

    original_basis_params = get_parameters(net, zero_grad=fix_basis, layer_name='basis')
    original_0_params = get_parameters(net, zero_grad=fix_basis, layer_name='{}0'.format(layer_type))
    original_weight_params = get_parameters(net, zero_grad=False, layer_name='weights')

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # full batch size == GD
    criterion = nn.MSELoss()

    trainer = Trainer(dataloaders=dataloaders, model=net, criterion=criterion, optimizer=optimizer,
                      record_rate=record_rate, print_rate=print_rate)

    trainer.train(num_epochs, verbose=False)

    # Plot and save losses
    train_loss = np.array(trainer.train_loss)
    val_loss = np.array(trainer.val_loss)
    weights_norms = np.array(list(map(lambda x: x, trainer.weights_norms)))
    print("final weights norms", weights_norms.shape)
    train_loss_file = 'train_loss_lr{}_netwidth{}_weight_std{}'.format(lr_str, net_width, weight_std)
    val_loss_file = 'val_loss_lr{}_netwidth{}_weight_std{}'.format(lr_str, net_width, weight_std)
    weights_norms_file = 'weights_norms_lr{}_netwidth{}_weight_std{}'.format(lr_str, net_width, weight_std)
    np.save(folder_name + train_loss_file, train_loss)
    np.save(folder_name + val_loss_file, val_loss)
    np.save(folder_name + weights_norms_file, weights_norms)
    if gdrive:
        files.download(gdrive_base_path+train_loss_file+'.npy')
        files.download(gdrive_base_path+val_loss_file+'.npy')

    if plot:
        losses_fig_file = 'losses_lr{}_netwidth{}_weight_std{}.png'.format(lr_str, net_width, weight_std)
        preds_fig_file = 'preds_lr{}_netwidth{}_weight_std{}.png'.format(lr_str, net_width, weight_std)
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

    return trainer, pred, train_dataset, test_dataset, dataloaders
