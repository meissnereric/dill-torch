from dill.experiment import run_experiment
import numpy as np
import shutil
data_folder='tst/test_outputs'
trainer1, pred, train1, test1 = run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                   lr_str="1e4", weight_decay=0, net_width=15, sigma=0.2,
                   num_epochs=100, plot=False, gdrive=False, weight_variance=0.01,
                   folder_name=data_folder+'/', rbfs=3, rbf_init='constant')

trainer1, pred, train1, test1 = run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                   lr_str="1e4", weight_decay=0, net_width=15, sigma=0.2,
                   num_epochs=100, plot=False, gdrive=False, weight_variance=0.01,
                   folder_name=data_folder+'/', rbfs=2, rbf_init='normal')

trainer1, pred, train1, test1 = run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                   lr_str="1e4", weight_decay=0, net_width=15, sigma=0.2,
                   num_epochs=100, plot=False, gdrive=False, weight_variance=0.01,
                   folder_name=data_folder+'/', rbfs=4, rbf_init='range')

shutil.rmtree(data_folder, ignore_errors=True)
