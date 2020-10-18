import torch

def batch_dot_product(a,b):
    b = b.T
    assert a.shape[1] == b.shape[0]
    return torch.matmul(a,b)
