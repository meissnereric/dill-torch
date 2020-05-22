import pytest
import shutil
import numpy as np
from torch.utils.data import DataLoader
from dill.experiment import run_experiment
from dill.simple_data import SinDataset


@pytest.mark.usefixtures("set_seed")
class TestTrainingEndToEnd():

    @pytest.mark.parametrize("train_samples, test_samples, learning_rate, net_width, sigma, num_epochs, weight_std, folder_name, hidden_layers, layer_type, record_rate", [
        (30, 300, 1e-4, 10, 0.2, 10*2, 1e-1,'tst/test_outputs/', 3, 'relu', 10),
        (30, 300, 1e-4, 10, 0.2, 10*2, 1e-1,'tst/test_outputs/', 1, 'relu', 10),
        (30, 300, 1e-4, 10, 0.2, 10*2, 1e-1,'tst/test_outputs/', 3, 'rbf', 10),
        (30, 300, 1e-4, 10, 0.2, 10*2, 1e-1,'tst/test_outputs/', 1, 'rbf', 10),
        ])
    def test_training(self, train_samples, test_samples, learning_rate, net_width, sigma, num_epochs, weight_std, folder_name, hidden_layers, layer_type, record_rate):
        trainer1, pred, train1, test1, data1 = run_experiment(train_samples=train_samples, test_samples=test_samples,
                                                              learning_rate=learning_rate, net_width=net_width, sigma=sigma,
                                                              num_epochs=num_epochs, plot=False, gdrive=False, weight_std=weight_std,
                                                              folder_name=folder_name, hidden_layers=hidden_layers, layer_type=layer_type,
                                                              record_rate=record_rate)
        shutil.rmtree(folder_name, ignore_errors=True)

    @pytest.mark.parametrize("train_samples, test_samples, learning_rate, net_width, sigma, num_epochs, weight_std, folder_name, hidden_layers, layer_type, record_rate", [
        (30, 300, 1e-4, 10, 0.2, 10*2, 1e-1,'tst/test_outputs/', 3, 'relu', 10),
        ])
    def test_training_custom_dataloaders(self, train_samples, test_samples, learning_rate, net_width, sigma, num_epochs, weight_std, folder_name, hidden_layers, layer_type, record_rate):

        train_dataset = SinDataset(train_samples)
        test_dataset = SinDataset(test_samples, std=0.0)
        train_loader = DataLoader(train_dataset, batch_size=train_samples)
        test_loader = DataLoader(test_dataset, batch_size=test_samples)
        dataloaders = {'train': train_loader, 'val': test_loader}

        trainer1, pred, train1, test1, data1 = run_experiment(train_samples=train_samples, test_samples=test_samples,
                                                              learning_rate=learning_rate, net_width=net_width, sigma=sigma,
                                                              num_epochs=num_epochs, plot=False, gdrive=False, weight_std=weight_std,
                                                              folder_name=folder_name, hidden_layers=hidden_layers, layer_type=layer_type,
                                                              record_rate=record_rate, dataloaders=dataloaders)
        shutil.rmtree(folder_name, ignore_errors=True)
