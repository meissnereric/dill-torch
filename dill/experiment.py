from .simple_data import SinDataset
from .simple_model import create_model, init_const_model
from .training import Trainer
from .utils import get_parameters
from .viz import visualize_predictions, make_predictions
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                   lr_str="1e4", weight_decay=0, net_width=30, sigma=0.2,
                   num_epochs=1000, plot=True):
    """
    Experimental code to test for double dip phenomenon.
    Batch size is always the full dataset so SGD == GD.

    """

    train_dataset = SinDataset(train_samples)
    test_dataset = SinDataset(test_samples, noise=0.0)
    train_loader = DataLoader(train_dataset, batch_size=train_samples)
    test_loader = DataLoader(test_dataset, batch_size=test_samples)
    dataloaders = {'train': train_loader, 'val': test_loader}

    net = create_model(net_width, sigma=sigma)
    init_const_model(net)

    original_basis_params = get_parameters(net, zero_grad=True, param_name='basis')
    original_rbf_params = get_parameters(net, zero_grad=True, param_name='rbf')

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # full batch size == GD
    criterion = nn.MSELoss()

    trainer = Trainer(dataloaders=dataloaders, model=net, criterion=criterion, optimizer=optimizer)

    trainer.train(num_epochs, verbose=False)

    # Plot and save losses
    train_loss = np.array(trainer.train_loss)
    val_loss = np.array(trainer.val_loss)
    with open('train_loss_lr{}_netwidth{}.txt'.format(lr_str, net_width), 'w') as f:
        f.write(str(train_loss))
    with open('val_loss_lr{}_netwidth{}.txt'.format(lr_str, net_width), 'w') as f:
        f.write(str(val_loss))

    if plot:
        pltlen = -1
        start=0
        plt.plot(train_loss[start:pltlen,0], [0] * len(train_loss[start:pltlen,0]), label='perfect loss')
        plt.plot(train_loss[start:pltlen,0], train_loss[start:pltlen,1], label='train loss')
        plt.plot(val_loss[start:pltlen,0], val_loss[start:pltlen,1], label='val loss')
        plt.legend()
        plt.savefig('losses_lr{}_netwidth{}.png'.format(lr_str, net_width))

        pred, fig = visualize_predictions(net, dataloaders)
        fig.savefig('preds_lr{}_netwidth{}.png'.format(lr_str, net_width))
    else:
        pred = make_predictions(net, dataloaders)

    return trainer, pred