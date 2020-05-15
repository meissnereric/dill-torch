from dill.experiment import run_experiment
import numpy as np
import shutil
data_folder='tst/test_outputs'
trainer1, pred, train1, test1, data1 = run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                   weight_decay=0, net_width=15, sigma=0.2,
                   num_epochs=10*2, plot=False, gdrive=False, weight_variance=0.01,
                   folder_name=data_folder+'/', hidden_layers=3, layer_type='relu',
                   record_rate=10)

trainer1, pred, train1, test1, data1 = run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                   weight_decay=0, net_width=15, sigma=0.2,
                   num_epochs=10*2, plot=False, gdrive=False, weight_variance=0.01,
                   folder_name=data_folder+'/', hidden_layers=2,
                   layer_type='rbf', init_type='normal', linear_hidden=True,
                   record_rate=10)

trainer1, pred, train1, test1, data1 = run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                  weight_decay=0, net_width=15, sigma=0.2,
                  num_epochs=10*2, plot=False, gdrive=False, weight_variance=0.01,
                  folder_name=data_folder+'/', hidden_layers=2,
                  layer_type='rbf', init_type='normal', linear_hidden=False,
                  record_rate=10)

trainer1, pred, train1, test1, data1 = run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                  weight_decay=0, net_width=15, sigma=0.2,
                  num_epochs=10*2, plot=False, gdrive=False, weight_variance=0.01,
                  folder_name=data_folder+'/', hidden_layers=1,
                  layer_type='rbf', init_type='normal', linear_hidden=False,
                  record_rate=10)

shutil.rmtree(data_folder, ignore_errors=True)
