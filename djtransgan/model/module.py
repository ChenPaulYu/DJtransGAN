import math
import torch
import torch.nn as nn

from djtransgan.utils import get_class_name
from djtransgan.model import init_weights, get_activate_func, get_norm_func, get_drop_func

# MLP

class MLP(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 drop_prob = None, 
                 activate  = None, 
                 norm_type = None):
        super(MLP, self).__init__()
        self.linear    = nn.Linear(in_dim, out_dim)
        self.dropout   = get_drop_func('drop1d', drop_prob) if drop_prob else None
        self.norm      = get_norm_func(norm_type, out_dim) if norm_type else None
        self.activate  = get_activate_func(activate) if activate else None
        self.apply(init_weights)
        
    def forward(self, x):
        x = self.linear(x)
        
        if self.dropout:
            x = self.dropout(x)
        if self.norm:
            if get_class_name(self.norm) == 'LayerNorm':
                norm = nn.LayerNorm(x.size()[1:]).to(x.device)
                x    = norm(x)
            else:
                x    = self.norm(x)
        
        if self.activate:
            x = self.activate(x)
    
        return x

# Conv

class Conv2d(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 k_size    = 3, # kernel_size
                 stride    = 1,                  
                 drop_prob = None, 
                 activate  = None, 
                 norm_type = None):
        
        super(Conv2d, self).__init__()
        self.conv     = nn.Conv2d(in_dim, 
                                 out_dim, 
                                 k_size, 
                                 stride  = stride, 
                                 padding = k_size//2)
        self.dropout  = get_drop_func('drop2d', drop_prob) if drop_prob else None
        self.activate = get_activate_func(activate) if activate  else None
        self.norm     = get_norm_func(norm_type, out_dim) if norm_type else None
        self.apply(init_weights)
        
    def forward(self, x):
        x = self.conv(x)
        
        if self.dropout:
            x = self.dropout(x)
        if self.norm:
            if get_class_name(self.norm) == 'LayerNorm':
                norm = nn.LayerNorm(x.size()[1:]).to(x.device)
                x    = norm(x)
            else:
                x    = self.norm(x)
        
        if self.activate:
            x = self.activate(x)
    
        return x