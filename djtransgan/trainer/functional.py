import torch
import auraloss
import torch.nn    as nn 
import torch.optim as optim

from djtransgan.config  import settings
from djtransgan.trainer import MinMaxLoss, LeastSquaresLoss

def get_optimizer(net, optim_name, lr, **kargs):
    
    optimizer_dict = {
        'SGD'    : optim.SGD(net.parameters()    , lr=lr, momentum=kargs.get('momentum', 0.9)), 
        'Adam'   : optim.Adam(net.parameters()   , lr=lr, betas=kargs.get('betas', (0., 0.999)), eps=kargs.get('eps', 1e-08)),
        'RMSProp': optim.RMSprop(net.parameters(), lr=lr, alpha=kargs.get('alpha', 0.99), eps=kargs.get('eps', 1e-08))
    }
    
    optimizer = optimizer_dict.get(optim_name)
    
    if optimizer is None:
        print(f'can not found {optim_name} loss')
        return 
    
    return optimizer

    
    
    
def get_criterion(loss_name):
    
    criterion_dict = {
        'minmax'      : MinMaxLoss(), 
        'least_square': LeastSquaresLoss()
    }
    
    criterion = criterion_dict.get(loss_name)
    
    if criterion is None:
        print(f'can not found {loss_name} loss')
        return 
    
    return criterion