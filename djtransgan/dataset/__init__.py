from djtransgan.config              import settings
from djtransgan.dataset.utils       import *
from djtransgan.dataset.noise       import *
from djtransgan.dataset.dataset     import *
from djtransgan.dataset.dataloader  import *
from djtransgan.dataset.datasampler import *

def get_dataset(data_type, **kargs):
    
    data_dict = {
        'noise': Noise(n_time    = kargs.get('n_time'   , settings.N_TIME), 
                       n_sample  = kargs.get('n_sample' , 100), 
                       norm_type = kargs.get('norm_type', 'loudness')
                      ), 
        'pair' : Pair(n_time     = kargs.get('n_time'    , settings.N_TIME), 
                      cue_ratio  = kargs.get('cue_ratio' , True), 
                      norm_type  = kargs.get('norm_type' , 'loudness')), 
        'mix'  : Mix(n_time      = kargs.get('n_time'    , settings.N_TIME), 
                      cue_ratio  = kargs.get('cue_ratio' , True), 
                      norm_type  = kargs.get('norm_type' , 'loudness'))
    }
    
    dataset   = data_dict.get(data_type)
    
    if dataset is None:
        print(f'can not found {data_type} dataset ...')
    
    return dataset