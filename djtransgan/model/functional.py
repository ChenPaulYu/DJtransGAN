import math
import torch
import torch.nn as nn

def get_activate_func(name):
    return {
        'tanh'   : nn.Tanh() ,
        'relu'   : nn.ReLU() ,
        'prelu'  : nn.PReLU(),
        'lrelu'  : nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(), 
    }.get(name)


def get_norm_func(name, out_dim, **kargs):
    n_group = kargs.get('group')
    return {
        'batch1d'   : nn.BatchNorm1d(out_dim),
        'batch2d'   : nn.BatchNorm2d(out_dim), 
        'layer'     : nn.LayerNorm(out_dim),
        'group'     : nn.GroupNorm(n_group if n_group else 1, out_dim)
    }.get(name)


def get_drop_func(name, prob):
    return {
        'drop1d'   : nn.Dropout(p=prob), 
        'drop2d'   : nn.Dropout2d(p=prob), 
    }.get(name)