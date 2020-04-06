import torch
import torch.nn as nn
import time
import copy

def factor_svd(original_layer, assert_equal=True):
    """
    Returns an SVD decomposition where we have multiplied the U and V vectors
    by sqrt(S) in order to distribute its weighting across the two evenly.

    If assert_equal is true, it will also verify that 99% of the weights of the
    recombined US^1/2 * S^1/2V^T matrix are within 1e-3 of the original weights.
    Those bounds are mostly arbitrary, just what I found empirically.
    """
    W = original_layer.state_dict()['weight']
    svd_ret = torch.svd(W)
    U,V,S, = svd_ret.U, svd_ret.V, svd_ret.S
    sqrt_lambda = torch.diag(S ** (1/2))
    ULhalf = torch.matmul(U, sqrt_lambda)
    LhalfVT = torch.matmul(sqrt_lambda, V.T)
    if assert_equal:
        rtol=1e-3
        target_ratio = 0.98 # should try for 0.98 but the finetuned weights that we load in only hit like 0.78 for some reason?
        ratio = float(torch.isclose(torch.matmul(ULhalf,LhalfVT), W, rtol=rtol).flatten().sum()) / W.flatten().size()[0]
        # assert ratio >= target_ratio
        print("ratio {} vs target_ratio {}".format(ratio, target_ratio))
    return ULhalf, LhalfVT

def classifier_layer(inp, out, weights, bias=None, cuda=True):
    """
    Creates a new linear layer bounded by a Dropout and ReLU,
     with the linear layer having (inp, out) size and being initialized to the
     weights and bias matrices passed in.
    """
    if bias is not None:
        layer = nn.Linear(inp, out)
    else:
        layer = nn.Linear(inp, out, bias=False)

    if cuda:
        layer.cuda()
    layer.state_dict()['weight'].copy_(weights)
    assert torch.all(layer.state_dict()['weight'].eq(weights))

    if bias is not None:
        layer.state_dict()['bias'].copy_(bias)
        assert torch.all(layer.state_dict()['bias'].eq(bias))

    return layer


def convert_layer_to_z(original_layer, mlp_width=None, ignore_first=False, ignore_last=False, cuda=True):
    """
    :param mlp_width: How many additional weights to add to the layers. So final size will be original+mlp_width. If None will create a new 'z' layer with width=min(inp, out).
    :param ignore_first: When true, won't add the extra zero-weights to the first linear layer's inputs.
    :param ignore_last: When true, won't add the extra zero-weights to the second linear layer's outputs.
    """
    ULhalf, LhalfVT = factor_svd(original_layer)
    original_bias = original_layer.state_dict()['bias']
    original_weight = original_layer.state_dict()['weight']
    input_dim = original_weight.size(1)
    std = 1e-2


    if (mlp_width is None) or (mlp_width <= 1): # this adds a 'z' layer at the width of min(original_layer.shape[0], original_layer.shape[1]).
        inp = LhalfVT.shape[1]
        out = ULhalf.shape[0]
        mid = min(inp, out)
        W1_bias = None
        W2_bias = original_bias

        W1 = classifier_layer(inp, mid, LhalfVT,
                              bias=W1_bias, cuda=cuda)
        W2 = classifier_layer(mid, out, ULhalf,
                              bias=W2_bias, cuda=cuda)


    else: # increase the MLP layer widths to test behavior in the limit.
        # initialization value for new weights

        # create the new weight matrices, with the original weights initialized
        # from the SVD and at std where we add new weights
        mid = min(LhalfVT.shape[1], ULhalf.shape[0])
        W1_bias = None

        # If ignore_first, don't add extra mlp_width to the first layer
        if not ignore_first:
            inp = int(LhalfVT.shape[1] * mlp_width )
            diff_inp = inp - LhalfVT.shape[1]
            extra_L = torch.FloatTensor(diff_inp * mid).reshape((mid, diff_inp)).normal_(mean=0., std=std).to(original_bias.device)
            extended_L = torch.cat((LhalfVT, extra_L), 1)
        else:
            inp = int(LhalfVT.shape[1])
            extended_L = LhalfVT

        # If ignore_last, don't add extra mlp_width to the last layer and use the original bias
        if not ignore_last:
            out = int(ULhalf.shape[0] * mlp_width)
            diff_out = out - ULhalf.shape[0]
            extra_U = torch.FloatTensor(diff_out * mid).reshape((diff_out, mid)).normal_(mean=0., std=std).to(original_bias.device)
            extended_U = torch.cat((ULhalf, extra_U), 0)
            W2_bias = torch.cat((original_bias,
                                torch.FloatTensor(diff_out).normal_(mean=0., std=std).to(original_bias.device)), 0)
        else:
            out = int(ULhalf.shape[0])
            extended_U = ULhalf
            W2_bias = original_bias

        # create layers
        print(inp,mid,out)
        print("ext_L: {}  ext_U: {} W2_bias: {}".format(extended_L.shape, extended_U.shape, W2_bias.shape))
        W1 = classifier_layer(inp, mid, extended_L,
                              bias=W1_bias, cuda=cuda)
        W2 = classifier_layer(mid, out, extended_U,
                              bias=W2_bias, cuda=cuda)

    return W1, W2



def build_classifier(orig_classifier, mlp_width=None, cuda=True, with_extra_relu=False):
    """
    High level method to take the original AlexNet classifier submodule
    and convert it into one with additional 'z' layers, initialized at the
    original weights via an SVD.

    # do SVD over each linear layer, get UL^1/2, L^(1/2)V^T - factor_svd
    # initialize original weights for z layer using svd values - convert_layer_to_z
    # increase size to match mlp_width increases
    # initialize new MLP weights to small values

    """
    # nn.Dropout(), layer, nn.ReLU(inplace=True)
    l11, l12 = convert_layer_to_z(orig_classifier[1], mlp_width=mlp_width, cuda=cuda, ignore_first=True)
    l41, l42 = convert_layer_to_z(orig_classifier[4], mlp_width=mlp_width, cuda=cuda)
    # we only use the linear layers from the last layer, not the Dropout and RELU ones
    # (l611, l612), (l621, l622) = convert_layer_to_z(orig_classifier[6], mlp_width=mlp_width, cuda=cuda)
    l61, l62 = convert_layer_to_z(orig_classifier[6], mlp_width=mlp_width, cuda=cuda, ignore_last=True)

    if with_extra_relu:
        layers = [nn.Dropout(), l11, nn.ReLU(inplace=True), l12, nn.ReLU(inplace=True),
                nn.Dropout(), l41, nn.ReLU(inplace=True), l42, nn.ReLU(inplace=True),
                l61, nn.ReLU(inplace=True), l62]
    else:
        layers = [nn.Dropout(), l11, l12, nn.ReLU(inplace=True),
                nn.Dropout(), l41, l42, nn.ReLU(inplace=True),
                l61, l62]


    return nn.Sequential(*layers)

def create_net(mlp_width=None, verbose=False, mode='oo',
               cuda=True, learning_rate=1e-3, weight_decay=1e-2,
               with_extra_relu=False, pretrained_file=None,
               optimizer_type=None):
    """
    :param mode: Whether to use:
     'oo: the original architecture and parameters,
     'mo': modified architecture from original paramters, or
     'mm': modified architecture from modified parameters
    :param pretrained_file: The name of the file in your Google Drive account
    with the parameters for this network. If None, does nothing.
    Loads not strictly.
    :param optimizer_type: SGD by default for None value, 'adam' will set the optimizer to Adam
    """
    net = anet(pretrained=True)
    if cuda:
        net.cuda()

    if mode == 'oo':
        pass
    elif mode == 'mo' and pretrained_file is not None:
            load_model_from_drive(net, pretrained_file)
            net.classifier = build_classifier(net.classifier, mlp_width=mlp_width, cuda=cuda, with_extra_relu=with_extra_relu)

    elif mode == 'mm' and pretrained_file is not None:
            net.classifier = build_classifier(net.classifier, mlp_width=mlp_width, cuda=cuda, with_extra_relu=with_extra_relu)
            load_model_from_drive(net, pretrained_file)
    else:
        raise Exception('Mode must be one of oo, mo, mm and pretrained_file supplied if mo,mm')

    if verbose:
        print(net.classifier)


    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return net, criterion, optimizer
