from dill.experiment import run_experiment
import numpy as np

_,_, train1, test1 = run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                   lr_str="1e4", weight_decay=0, net_width=15, sigma=0.2,
                   num_epochs=100, plot=False, gdrive=False, weight_noise=0.01)


_, _, train2, test2 = run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                  lr_str="1e4", weight_decay=0, net_width=10, sigma=0.2,
                  num_epochs=10, plot=False, gdrive=False, weight_noise=0.1)

print("Train datasets equal: ", np.all(np.equal(train1, train2)))
print("Test datasets equal: ", np.all(np.equal(test1, test2)))
