from djtransgan.model.utils         import *
from djtransgan.model.functional    import *
from djtransgan.model.module        import *
from djtransgan.model.block         import *
from djtransgan.model.component     import *
from djtransgan.model.generator     import *
from djtransgan.model.discriminator import *


def get_generator(cnn_type='res_cnn'): # loss_type
    
    encoder_args   = {
        'in_dim'   : 1         , 
        'out_dims' : [4, 8, 16],
        'last_dim' : None      ,
        'cnn_type' : cnn_type  , 
        'activate' : 'relu'    ,
        'norm_type': 'batch2d' 
    }

    processor_args = {
        'out_dims' : [1024, 512], 
        'activate' : 'lrelu',
        'norm_type': 'batch1d', 
        'pool_type': 'mlps'
    }

    context_args = {
        'activate' : 'lrelu', 
        'norm_type': 'batch2d'
    }

    unzipper_args  = {
        'n_band' : 4, 
        'n_fader': 1, 
        'n_param': 2
    }

    mixer_args     = {
        'band_args'  : {
            'split_type': 'bias'  ,
            'out_type'  : 'track' ,
            'sum_type'  : 'mean'  , 
            'band_type' : 'low'   , 
            'fade_shape': 'linear'
        }, 
        'fader_args' : {
            'sum_type'  : 'mean'  ,
            'fade_shape': 'linear'
        }
    }
    
    return Generator(unzipper_args, 
                     encoder_args, 
                     context_args, 
                     processor_args, 
                     mixer_args)

def get_discriminator(cnn_type, loss_type):
    
    encoder_args = {
        'in_dim'   : 1, 
        'out_dims' : [4, 8, 16], 
        'last_dim' : None,
        'cnn_type' : cnn_type, 
        'activate' : 'lrelu', 
        'norm_type': 'batch2d'
     }

    processor_args = {
        'out_dims' : [1024, 512],
        'activate' : 'lrelu',
        'norm_type': 'batch1d', 
        'pool_type': 'mlps',
        'loss_type': loss_type
    }
    
    return Discriminator(encoder_args, 
                         processor_args)


def get_net(**kargs):
    cnn_type  = kargs.get('cnn_type' , 'res_cnn')
    loss_type = kargs.get('loss_type', 'minmax') 
    
    G = get_generator(cnn_type)
    D = get_discriminator(cnn_type, loss_type)
    return (G, D)
