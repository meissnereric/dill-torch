import torch.nn.init as init
import torch.nn as nn

net = nn.Sequential()
net.add_module('basis', nn.Linear(1, net_width, bias=False))
# net.add_module('relu', nn.ReLU())
net.add_module('rbf', RBF(net_width, net_width, gaussian))
net.add_module('weights', nn.Linear(net_width, 1, bias=False))


apply_init(net, 'basis', constant_init)
apply_init(net, 'weights', constant_init)

output = net(torch.Tensor([[0.1],[1]]))
print('*************** Test outputs for x = [[0.1], [1.]] ************************')
print(output)
plt.plot(output[0].detach().numpy())
plt.plot(output[1].detach().numpy())
