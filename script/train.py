import warnings
warnings.filterwarnings('ignore')

import os
import sys
if os.getcwd() in sys.path:
    sys.path.append('../')
else:    
    sys.path.append(os.getcwd())
    
import torch
import argparse

from djtransgan.config       import settings
from djtransgan.utils        import check_exist, out_json
from djtransgan.dataset      import get_dataset
from djtransgan.model        import get_net
from djtransgan.trainer      import get_trainer



torch.manual_seed(settings.RANDOM_SEED)


def main():
    
    parser  = argparse.ArgumentParser(description='GAN Trainer')
    parser.add_argument('--n_gpu'     , default = 0)
    parser.add_argument('--epoch'     , default = 3)
    parser.add_argument('--log'       , default = 5)
    parser.add_argument('--lr'        , type=float, nargs='+')
    parser.add_argument('--n_critic'  , default = 1)
    parser.add_argument('--batch_size', type=int  , default=4)
    parser.add_argument('--loss_type' , default = 'minmax')
    parser.add_argument('--cnn_type'  , type=str , default='res_cnn')
    parser.add_argument('--out_dir'   , type=str , default='gan')
    parser.add_argument('--n_sample'  , type=int , default='4')
     
     
    args     = parser.parse_args()
    arg_dict = vars(args)
    
    print('load dataset begin...')
    dataset = [get_dataset('pair'), get_dataset('mix')]
    print('load dataset end...')
    
    print('load net begin...')
    net     = get_net(cnn_type=args.cnn_type, loss_type=args.loss_type)
    print('load net end...')
    
    print('load trainer begin...')
    trainer = get_trainer(net, 
                          dataset, 
                          log        = args.log,
                          lr         = args.lr,
                          n_gpu      = args.n_gpu,
                          epoch      = args.epoch, 
                          n_critic   = args.n_critic, 
                          batch_size = args.batch_size, 
                          loss_type  = args.loss_type,
                          out_dir    = args.out_dir
                         )
    print('load trainer end...')
    # save argument
    out_path = os.path.join(trainer.out_dir, 'args.json')
    check_exist(out_path)
    out_json(vars(args), out_path)
    
    trainer.train()
    
    

if __name__ == "__main__":
    main()
    
