import math
import torch
import torch.nn.functional as F


def purify_device(data, idxs=None):    
    if idxs is None: 
        idxs = range(len(data))
    if isinstance(data, list):        
        return [d.detach().cpu().clone() if i in idxs else d for i, d in enumerate(data)]
    if isinstance(data, tuple):
        return tuple([d.detach().cpu().clone() if i in idxs else d for i, d in enumerate(data)])
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().clone()
    
def unsqueeze_dim(data, num):
    for i in range(num):
        data = data.unsqueeze(-1)
    return data

def squeeze_dim(data):
    dims = [i for i in range(len(data.size())) if data.size(i) == 1]
    for dim in dims:
        data = data.squeeze(dim)
    return data