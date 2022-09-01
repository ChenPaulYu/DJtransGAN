import math
import copy
import torch
import torchaudio
import torch.nn as nn

from functools            import reduce
from djtransgan.config    import settings  
from djtransgan.model     import init_weights, estimate_channel, estimate_cnn_out_dim, get_activate_func
from djtransgan.model     import MLP, Conv2d
from djtransgan.model     import Conv2dBlock, ResConv2dBlock, PoolingBlock
from djtransgan.frontend  import get_amp2db_func, get_stft_func, get_mel_func

#### Fudemental Component ####

# CNN Models

class CNN(nn.Module):
    def __init__(self, 
                 in_dim      = 1, 
                 out_dims    = [4, 8, 16],
                 kernel_size = 3,
                 last_dim    = None   , # last dim is provied for poolblock to compress it to specific size
                 activate    = 'lrelu',
                 norm_type   = 'batch2d'):
        super(CNN, self).__init__()
        
        self.n_down        = len(out_dims)
        self.n_layer       = len(out_dims) + 1
        in_dims            = [in_dim] + out_dims[:-1]
        conv_blocks        = [Conv2dBlock(in_dim, 
                                          out_dim,
                                          k_size    = k_size,
                                          activate  = activate, 
                                          norm_type = norm_type) for (in_dim, out_dim) in zip(in_dims, out_dims)]
        
        conv_blocks       += [Conv2d(out_dims[-1], 1, k_size=1, activate=activate, norm_type=norm_type)]
        self.conv_blocks   = nn.ModuleList(conv_blocks)
        
        if last_dim:
            out_dim   = estimate_cnn_out_dim(self.ndown, last_dim)
            self.pool = PoolingBlock((None, out_dim), pool_type='avg')
        else:
            self.pool = None
        
        self.apply(init_weights)

        
    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            
        if self.pool:
            x = self.pool(x) 
        return x
    
class ResCNN(nn.Module):
    def __init__(self, 
                 in_dim      = 1, 
                 out_dims    = [4, 8, 16], 
                 last_dim    = None      , # last dim is provied for poolblock to compress it to specific size
                 k_size      = 3         ,
                 activate    = 'lrelu'   , 
                 norm_type   = 'batch2d'):
        
        super(ResCNN, self).__init__()
        
        self.n_down        = len(out_dims)
        self.n_layer       = len(out_dims) + 1
        in_dims            = [in_dim] + out_dims[:-1]
        conv_blocks        = [ResConv2dBlock(in_dim, 
                                             out_dim, 
                                             k_size      = k_size,
                                             activate    = activate,
                                             norm_type   = norm_type) for (in_dim, out_dim) in zip(in_dims, out_dims)]
        conv_blocks       += [Conv2d(out_dims[-1], 1, k_size=1, activate=activate, norm_type=norm_type)]
        self.conv_blocks  = nn.ModuleList(conv_blocks)
        
        if last_dim:
            out_dim   = estimate_out_dim(net_type='cnn', 
                                         last_dim=last_dim, 
                                         n_down=self.ndown)
            self.pool = PoolingBlock((None, out_dim), pool_type='avg')
        else:
            self.pool = None
        
        self.apply(init_weights)

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            
        if self.pool:
            x = self.pool(x) 
        return x
    
    
    
class MLPs(nn.Module):
    def __init__(self, 
                 in_dim    = 1, 
                 out_dims  = [512, 256],
                 activate  = 'lrelu', 
                 norm_type = 'batch1d'):
        
        super(MLPs, self).__init__()
        
        self.n_layer    = len(out_dims)+2
        in_dims         = [in_dim] + out_dims[:-1]
        mlps            = [MLP(in_dim, 
                               out_dim, 
                               activate  = activate, 
                               norm_type = norm_type) for (in_dim, out_dim) in zip(in_dims[:-1], out_dims[:-1])]
        mlps           += [MLP(in_dims[-1], out_dims[-1])]
        self.mlps       = nn.ModuleList(mlps)
        self.apply(init_weights)

    def forward(self, x):
        if len(x.size()) > 2:
            x = torch.flatten(x, start_dim=1)
            
        for i, mlp in enumerate(self.mlps):
            x = mlp(x)            
        return x    
    
#### Advanced Component ####

class Frontend(nn.Module):
    def __init__(self, 
                 stft_type = 'torchlibrosa', 
                 power     = 1, 
                 length    = None):
        super(Frontend, self).__init__()
        
        self.power     = power
        self.stft_type = stft_type
        self.mel_scale = get_mel_func()
        self.stft      = get_stft_func(stft_type = stft_type, 
                                       power     = power, 
                                       length    = length)
        self.amp2db    = get_amp2db_func(power=power)
        
    def mel_sacle(self, mag):
        mel_mag = self.mel_scale.to(mag.device)(mag)
        mel_mag = self.amp2db.to(mel_mag.device)(mel_mag)
        return mel_mag
    
    def inverse(self, mags, phases, length=None):
        return self.stft.to(mags.device).inverse(mags, phases, length)
        
    def forward(self, x, phase=False):
        out_tuple = ()
        if phase:
            mag, phase = self.stft.to(x.device)(x, phase=phase)
            out_tuple += (mag, phase)
        else:
            if len(x.size()) < 4:
                mag,  = self.stft.to(x.device)(x, phase=phase)
            else:
                mag   = x  
            out_tuple = (mag,)
        mel_mag   = self.mel_sacle(mag)
        out_tuple = (mel_mag,) + out_tuple 
        return out_tuple
    
    
class Encoder(nn.Module):
    def __init__(self, encoder_args):
        super(Encoder, self).__init__()
        
        encoder_args = copy.deepcopy(encoder_args)
        
        if 'times' in encoder_args.keys():
            encoder_args.pop('times')

        cnn_type     = encoder_args.pop('cnn_type')
        net          = {'cnn': CNN, 'res_cnn': ResCNN}[cnn_type]
        
        self.net     = net(**encoder_args)
        self.apply(init_weights)
        
        self.n_layer = self.net.n_layer
        self.n_down  = self.net.n_down
        
        
    def forward(self, x):
        return self.net(x)


class Context(nn.Module):
    def __init__(self, 
                 activate  = 'lrelu', 
                 norm_type = 'batch2d',
                ):
        super(Context, self).__init__()
        self.conv = Conv2d(2,
                           1,
                           k_size    = 1, 
                           activate  = activate, 
                           norm_type = norm_type)
        self.apply(init_weights)
        
    def forward(self, x1, x2):
        if len(x1.size()) == 3:
            y = torch.stack((x1, x2), axis=1)
        else:
            y = torch.cat((x1, x2), axis=1)
        y = self.conv(y)
        return y
    

class PostProcessor(nn.Module):
    def __init__(self, 
                 processor_args, 
                 context_args=None):
        
        super(PostProcessor, self).__init__()
        
        processor_args   = copy.deepcopy(processor_args)
        self.pool_type   = processor_args.pop('pool_type')
        self.context     = Context(**context_args) if context_args else None
        
        if 'last_activate' in processor_args.keys():
            last_activate = processor_args.pop('last_activate')
        else:
            last_activate = None
            
        if 'loss_type' in processor_args.keys():
            processor_args.pop('loss_type')
            
        if self.pool_type:
            self.predictor = MLPs(**processor_args)
        else:
            if 'norm' in processor_args.keys():
                processor_args.pop('norm')
            self.predictor = PoolingBlock(**processor_args)
    
        self.last_activate = get_activate_func(last_activate)
        
    def forward(self, x, z=None):
        
        if self.context and z is not None:
            x = self.context(x, z)     

        y = self.predictor(x)
        if self.last_activate:
            y = self.last_activate(y)
            
        return y
    
    
class Unzipper(nn.Module):
    def __init__(self, n_band=4, n_fader=1, n_param=2):
        
        super(Unzipper, self).__init__()
        self.shape     = {'band':  ((n_band - 1), 2), 
                          'fader': (n_band, n_fader, n_param)}    
        self.n_params  = {k: reduce(lambda x, y: x*y, v)  for (k, v) in self.shape.items()}
        self.s_param   = reduce(lambda x, y: x+y, self.n_params.values()) # total num of parameters
        
    def forward(self, params):
        assert params.size(-1) == self.s_param
        unzip_params = {}

        if self.n_params['band'] == 0:
            unzip_params['fader'] = params
        elif self.n_params['fader'] == 0:
            unzip_params['band']  = params
        else:
            (band_params, fader_params) = torch.split(params, tuple(self.n_params.values()), dim=-1)
            unzip_params['band']  = band_params
            unzip_params['fader'] = fader_params
            
        return {k: unzip_params[k].view((-1,) + self.shape[k]) for k in unzip_params}