import math
import torch
import torch.nn as nn

from djtransgan.model import get_activate_func, get_norm_func
from djtransgan.model import Conv2d



class Conv2dBlock(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 k_size      = 3,     
                 stride      = 2,
                 activate    = 'lrelu',
                 norm_type   = 'batch2d'):
        super(Conv2dBlock, self).__init__()
        
        strides    = [stride, 1]
        in_dims    = [in_dim, out_dim]
        self.convs = nn.ModuleList([Conv2d(in_dim, 
                                           out_dim, 
                                           k_size    = k_size,
                                           stride    = stride, 
                                           activate  = activate, 
                                           norm_type = norm_type)
                                           for (in_dim, stride) in zip(in_dims, strides)])
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
    
    
class ResConv2dBlock(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 k_size    = 3, 
                 stride    = 2,
                 activate  = 'relu',
                 norm_type = 'batch2d'):
    
        super(ResConv2dBlock, self).__init__()
        strides       = [stride, 1, stride]
        in_dims       = [in_dim, out_dim, in_dim]
        self.convs    = nn.ModuleList([Conv2d(in_dim, 
                                              out_dim, 
                                              k_size    = k_size,
                                              stride    = stride,
                                              activate  = activate,
                                              norm_type = norm_type) 
                                       for (in_dim, stride) in zip(in_dims, strides)])
        self.diff     = (strides[-1] != 1) or (in_dim != out_dim)
        self.activate = get_activate_func(activate)
    
    def forward(self, x):
        y      = x.clone()
        for conv in self.convs[:2]:
            y = conv(y)
            
        if self.diff:
            x = self.convs[-1](x)
        
        y = x + y
        y = self.activate(y)
        return y
    
class PoolingBlock(nn.Module):
    def __init__(self, 
                 out_dim   = 1024, # tuple -> 2D, int -> 1D
                 activate  = None, 
                 pool_type = 'avg'):
        super(PoolingBlock, self).__init__()
        
        if isinstance(out_dim, tuple):
            self.data_type = '2d'
            self.pool      = {'avg': nn.AdaptiveAvgPool2d, 'max': nn.AdaptiveMaxPool2d}[pool_type](out_dim)
        else:
            self.data_type = '1d'
            self.pool      = {'avg': nn.AdaptiveAvgPool1d, 'max': nn.AdaptiveMaxPool1d}[ptype](out_dim)  
        
        self.activate      = get_activate_func(activate)
        
    def foward(self, x):
        if self.data_type == '1d':
            y = torch.flatten(x, start_dim=1).unsqueeze(1)
            y = self.pool(y).squeeze(1)
        else:
            y = self.pool(x)
            
        if self.activate:
            y = self.activate(y)
        return y
            
            
    
