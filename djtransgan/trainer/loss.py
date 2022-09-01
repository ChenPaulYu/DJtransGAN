import torch
import torch.nn as nn

from djtransgan.config import settings
from djtransgan.utils  import reduce

def get_labels(data, true=False):
    return torch.ones_like(data) if true else torch.zeros_like(data)


class MinMaxLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MinMaxLoss, self).__init__()
        self.reduction  = reduction
        self.criterion  = nn.BCELoss(reduction=reduction)
        
    def generator_loss(self, dgz):
        reals = get_labels(dgz, true=True)
        return self.criterion(dgz, reals)
        
        
    def discriminator_loss(self, 
                           dgz, 
                           dx):
        fakes      = get_labels(dgz, true=False)
        reals      = get_labels(dx , true=True)
        fake_loss  = self.criterion(dgz, fakes)
        real_loss  = self.criterion(dx , reals)
        return fake_loss + real_loss
        
        
    def forward(self, dgz, dx=None):
        if dx is None:
            return self.generator_loss(dgz)
        return self.discriminator_loss(dgz, dx)
    
    
class LeastSquaresLoss(nn.Module):
    def __init__(self, a=0.0, b=1.0, c=1.0, reduction='mean'):
        super(LeastSquaresLoss, self).__init__()
        self.reduction  = reduction
        self.a = a
        self.b = b
        self.c = c
    
    
    def generator_loss(self, dgz):
        return 0.5 * reduce((dgz - self.c) ** 2, self.reduction)
        
        
    def discriminator_loss(self, dgz, dx):
        fake_loss  = 0.5 * (reduce((dgz - self.a) ** 2, self.reduction))
        real_loss  = 0.5 * (reduce((dx - self.b) ** 2, self.reduction))
        return fake_loss + real_loss
        
        
    def forward(self, dgz, dx=None):
        if dx is None:
            return self.generator_loss(dgz)
        import pdb;pdb
        return self.discriminator_loss(dgz, dx)