def make_layers_for_testing(original_linear, verbose=True, with_extra_relu=False):
    """
    Makes a new layer without a dropout input,
    and appends a RELU to both old and new for testing.
    """
    l11, l12 = convert_layer_to_z(original_linear, mlp_width=None, cuda=True)
    orig_layer = nn.Sequential(original_linear, nn.ReLU(inplace=True))
    if with_extra_relu:
        new_layer = nn.Sequential(l11, nn.ReLU(inplace=True), l12, nn.ReLU(inplace=True))
    else:
        new_layer = nn.Sequential(l11, l12, nn.ReLU(inplace=True))
    if verbose:
        print(orig_layer, new_layer)

    return orig_layer, new_layer

def test_equal_layers(net, new_net, data=None, first_linear_layer=0, verbose=False):
    """
    Unit test function for convert_layer_to_z(...)
    I don't expect all inputs to match exactly, but use the sum of all
     computed data points as a proxy to measure against.

     :param first_linear_layer: The index of the first layer in the network
    that has input_features. Used to get the right size for the data creation.

    """
    net.cuda()
    new_net.cuda()
    data = data if data is not None else torch.FloatTensor(1000, net[first_linear_layer].in_features).normal_(0, 5).to('cuda:0')
    outputs = net(data)
    new_outputs = new_net(data)

    rtol=1e-3
    target_ratio = 0.99
    ratio = float(torch.isclose(new_outputs, outputs, rtol=rtol).flatten().sum()) / outputs.flatten().size()[0]
    print(ratio)
    print("abs diff: ", abs(new_outputs - outputs).sum())
    assert ratio >= target_ratio


def compute_gradient_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_nor

def set_dropout(net, p=0.5):
    """
    Sets the dropout values of the children of the net's classifier.
    The default for AlexNet is 0.5.
    Used to set them to 0 so that there is no randomness when testing.
    """
    for c in net.classifier.children():
        if isinstance(c, nn.Dropout):
            c.p = p

def test_networks_equal(original_net, modified_net):
    set_dropout(original_net, p=0.0)
    set_dropout(modified_net, p=0.0)
    test_equal_layers(*make_layers_for_testing(original_net.classifier[1], verbose=False))
    test_equal_layers(*make_layers_for_testing(original_net.classifier[4], verbose=False))
    test_equal_layers(*make_layers_for_testing(original_net.classifier[6], verbose=False))

    test_equal_layers(original_net.classifier, modified_net.classifier, first_linear_layer=1, verbose=False)
    set_dropout(original_net, p=0.5)
    set_dropout(modified_net, p=0.5)
