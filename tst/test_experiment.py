from dill.experiment import run_experiment

run_experiment(train_samples=30, test_samples=300, learning_rate=1e-4,
                   lr_str="1e4", weight_decay=0, net_width=30, sigma=0.2,
                   num_epochs=1000, plot=False)
