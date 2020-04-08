
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_outputs(outputs, final_layer=False, verbose=False):
    """
    Accepts a torch.Tensor and plots it.
    """
    if verbose:
        print('***************************************')
        print(outputs)
    if not final_layer:
        for sample in outputs.T:
            sample = sample.detach().numpy()
            plt.plot(sample)
    else:
        plt.plot(outputs.detach().numpy())
    if verbose:
        print('***************************************')

def make_predictions(net, dataloaders):
    predictions = []
    for (inp, label) in dataloaders['val']:
        inputs = inp.reshape(inp.shape[0],1)
        pred = net(inputs).reshape(inp.shape[0])
        predictions.append([inp.numpy(), label.numpy(), pred.detach().numpy()])
    predictions = np.array(predictions)

    print("Predictions: {}".format(predictions))
    return predictions

def visualize_predictions(net, dataloaders):
    predictions = make_predictions(net, dataloaders)
    ax = plt.axes()
    ax.plot(predictions[0][0], predictions[0][1], label='true function')
    ax.plot(predictions[0][0], predictions[0][2], label='predictions')
    ax.legend()

    return predictions, plt.gcf()
