
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_outputs(outputs, final_layer=False):
    """
    Accepts a torch.Tensor and plots it.
    """
    print('***************************************')
    print(outputs)
    if not final_layer:
        for sample in outputs:
            sample = sample.detach().numpy()
            plt.plot(sample)
    else:
        plt.plot(outputs.detach().numpy())
    print('***************************************')
