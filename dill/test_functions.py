import torch
import torch.nn as nn

test_tensor = torch.Tensor([[0.1],[1]])

# What is the output at layer 1
def plot_outputs(outputs):
    print('***************************************')
    print(outputs)
    for sample in outputs:
        sample = sample.detach().numpy()
        plt.plot(sample)
    print('***************************************')

def test_layer_one_outputs(data):
    one_layer = nn.Linear(1, net_width, bias=False)
    init.constant_(one_layer.weight.data, 1)
    output = one_layer(data)
    plot_outputs(output)

# What is the output at layer 2
def test_layer_two_outputs(data):
    two_layer = nn.Sequential()
    two_layer.add_module('basis', nn.Linear(1, net_width, bias=False))
    # net.add_module('relu', nn.ReLU())
    two_layer.add_module('rbf', RBF(net_width,net_width,gaussian))
    apply_init(two_layer, 'basis', constant_init)
    output = two_layer(data)
    plot_outputs(output)


# What is the final output at layer 3
def test_layer_three_outputs(data):
    net = nn.Sequential()
    net.add_module('basis', nn.Linear(1, net_width, bias=False))
    # net.add_module('relu', nn.ReLU())
    net.add_module('rbf', RBF(net_width,net_width,gaussian))
    net.add_module('weights', nn.Linear(net_width, 1, bias=False))


    apply_init(net, 'basis', constant_init)
    apply_init(net, 'weights', constant_init)

    output = net(data)
    print('***************************************')
    print(outputs)
    plt.plot(output.detach().numpy())
    print('***************************************')

def visualize_predictions(net, dataloaders):
    predictions = []
    for (inp, label) in dataloaders['val']:
        inputs = inp.reshape(inp.shape[0],1)
        pred = net(inputs).reshape(inp.shape[0])
        predictions.append([inp.numpy(), label.numpy(), pred.detach().numpy()])
    predictions = np.array(predictions)

    print("Predictions: {}".format(predictions))

    ax = plt.axes()
    ax.plot(predictions[0][0], predictions[0][1], label='true function')
    ax.plot(predictions[0][0], predictions[0][2], label='predictions')
    ax.legend()

    return predictions
