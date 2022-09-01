import copy
import torch
import torch.nn as nn

from djtransgan.model import estimate_postprocessor_in_dim, estimate_postprocessor_out_dims
from djtransgan.model import Encoder, Frontend, PostProcessor
     
class Discriminator(nn.Module):
    
    def __init__(self, 
                 encoder_args, 
                 processor_args):
        
        super(Discriminator, self).__init__()
        
        self.hyper           = self.get_hyperparameter(encoder_args, processor_args)
        
        self.frontend        = Frontend() 
        encoder_args         = copy.deepcopy(encoder_args)
        
        processor_args['in_dim']        = estimate_postprocessor_in_dim(encoder_args)
        processor_args['out_dims']      = processor_args.get('out_dims') + [2 if processor_args.get('loss_type') == 'minmax' else 1]
        processor_args['last_activate'] = 'sigmoid' if processor_args.get('loss_type') == 'minmax' else None 
        self.post_processor  = PostProcessor(processor_args)
        self.encoder         = Encoder(encoder_args)

        
    def get_hyperparameter(self, encoder_args, processor_args):
        return {
                'encoder_args'  : copy.deepcopy(encoder_args), 
                'processor_args': copy.deepcopy(processor_args)
               }
        
    def encode(self, in_data):
        return self.encoder(self.frontend(in_data)[0])
                
    def forward(self, in_data):
        return self.post_processor(self.encode(in_data))
