import torch
from torchvision.datasets import MNIST



def load_data(ratio=1):
    tr_ds = MNIST('mnist', True, None, download=True)
    n_sample = len(tr_ds)
    idx = torch.randperm(n_sample)[:int(n_sample * ratio)]
    tr_ds = ((tr_ds.data[:, None, ...] / 255.)[idx], tr_ds.targets[idx])
    test_ds = MNIST('mnist', False, None, download=True)
    test_ds = (test_ds.data[:, None, ...] / 255., test_ds.targets)
    
    return tr_ds, test_ds
