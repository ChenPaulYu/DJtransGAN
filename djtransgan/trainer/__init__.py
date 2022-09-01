from djtransgan.trainer.loss       import *
from djtransgan.trainer.functional import *
from djtransgan.trainer.storer     import *
from djtransgan.trainer.trainer    import *



def get_trainer(net, dataset, **kargs):
    
    loss_type = kargs.get('loss_type', 'minmax')
    lr        = kargs.get('lr') 
    
    if len(dataset) != 2:
        print('please provide proper dataset ...')
        return
    
    if len(net) != 2:
        print('please provide proper G and D ...')
        return
    
    model_args = {
        'G': net[0], 
        'D': net[1]
    }
    
    train_args = {
        'lr'        : [1e-5, 1e-5] if lr is None else lr ,
        'optim'     : ['Adam', 'Adam'], 
        'epoch'     : kargs.get('epoch'     ,  5), 
        'n_critic'  : kargs.get('n_critic'  ,  1),
        'batch_size': kargs.get('batch_size',  4), 
        'n_gpu'     : kargs.get('n_gpu'     , -1), 
        'loss_type' : loss_type, 
        'dataset'   : dataset
    }
    
    store_args = {
        'n_sample'    : kargs.get('n_sample', 4) ,
        'log_interval': kargs.get('log'     , 20), 
        'out_dir'     : kargs.get('out_dir' , f'{loss_type}_gan')
    }
    
    return Trainer(model_args, train_args, store_args)