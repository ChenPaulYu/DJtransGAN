import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn          import Parameter
from functools         import wraps, reduce
from djtransgan.config import settings


# Estimate Dimentions

def estimate_channel(n_down):
    step  = 2**(n_down)
    return int(settings.N_MELS // step)

def estimate_frame(in_dim, n_down, in_type):
    step       = 2**n_down
    frame_dict = {'time'   : round(in_dim*settings.SR/settings.HOP_LENGTH), 
                  'sample' : round(in_dim/settings.HOP_LENGTH), 
                  'frame'  : in_dim
                 }
    frame      = frame_dict.get(in_type)
    
    if frame is None:
        print(f'do not found {in_type} input type')
    
    return int(math.ceil(frame/step))

def estimate_cnn_out_dim(n_down, last_dim):
    return int(last_dim/estimate_channel(n_down))

def estimate_mlps_in_dim(in_dim, n_down, in_type='time'):
    out_dims = (estimate_channel(n_down), 
                estimate_frame(in_dim, n_down, in_type))
    out_dim  = reduce(lambda x, y: x*y, out_dims) 
    return (out_dim, out_dims)

def estimate_postprocessor_in_dim(encoder_args):
    in_dim = encoder_args.get('last_dim')
    if in_dim is None:
        in_dim = estimate_mlps_in_dim(encoder_args.get('n_time', settings.N_TIME), 
                                     len(encoder_args.get('out_dims')))[0]
    return in_dim

def estimate_postprocessor_out_dims(processor_args, unzipper_args):
    n_band   = unzipper_args['n_band']
    n_fader  = unzipper_args['n_fader']
    n_param  = unzipper_args['n_param']
    last_dim = (n_band-1) * 2 + n_band * n_fader * n_param
    out_dims = [last_dim]
    if processor_args.get('out_dims'):
        out_dims = processor_args.get('out_dims') + out_dims
    return out_dims

    
# Inititalize

def init_weights(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
                
# Others

    
def empty_cache():
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
def freezing(net, action='freeze'): # e.g: freeze, unfreeze    
    for p in net.parameters():
        p.requires_grad = action == 'freeze'