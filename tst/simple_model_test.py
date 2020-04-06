from dill.test_functions import test_full_simple_model
import torch

test_tensor = torch.Tensor([[0.1],[1]])
test_full_simple_model(test_tensor)
