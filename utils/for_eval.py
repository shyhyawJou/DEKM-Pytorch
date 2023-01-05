from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np

import torch
from torch import nn



def evaluate(model, ds, batch_size, device):
    model.eval()
    mse = nn.MSELoss()
    n_sample = ds.shape[0]
    n_batch = n_sample // batch_size if n_sample % batch_size == 0 else n_sample // batch_size + 1
    
    loss = 0.    
    with torch.no_grad():
        for i in range(n_batch):
            x = ds[i * batch_size : (i + 1) * batch_size]
            x = x.to(device)
            _, gen = model(x)
            loss += mse(x, gen) * x.shape[0]
            print(f'{i + 1}/{n_batch}', end='\r')
    
    return loss / n_sample


def accuracy(truth, predict):
    confusion_m = confusion_matrix(truth, predict)
    _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    acc = np.trace(confusion_m[:, col_idx]) / confusion_m.sum()
    return acc


