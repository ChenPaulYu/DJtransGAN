import torch
import torch.nn as nn

from djtransgan.config     import settings
from djtransgan.mixer.mask import VMask, HMask

class Mixer(nn.Module):
    def __init__(self, band_args, fader_args):
        super(Mixer, self).__init__()
        self.vmask     = self.build_vmask(band_args)
        self.hmask     = self.build_hmask(fader_args)
        self.out_type  = band_args['out_type']
        
    def build_vmask(self, band_args):        
        return VMask(band_args['sum_type'], 
                     band_args['band_type'], 
                     band_args['fade_shape'], 
                     band_args['split_type'])
    
    def build_hmask(self, fader_args):
        return HMask(fader_args['sum_type'], 
                     fader_args['fade_type'],  
                     fader_args['fade_shape'])
    
    def _get_cue_region(self, cue_region):
        return torch.tensor([0, 1]) if cue_region is None else cue_region
    
    def forward(self, 
                mags, 
                band_params=None, 
                fader_params=None, 
                cue_region=None):
        out_dict = {}
        if band_params is not None:
            processed_mags, band_masks, _, band_curves = self.vmask(mags, band_params)
            out_dict['band'] = band_curves
            processed_masks  = band_masks
        else:
            processed_mags  = mags.unsqueeze(1)
            processed_masks = torch.ones_like(processed_mags).cpu()
        
        if fader_params is not None:
            cue_region  = self._get_cue_region(cue_region).to(mags.device)
            processed_mags, fader_masks, fader_curves = self.hmask(processed_mags, 
                                                                   fader_params, 
                                                                   cue_region=cue_region)
            out_dict['fader'] = fader_curves
            processed_masks  *= fader_masks
             
        out_dict['mask'] = processed_masks
        
        if processed_mags.size(1) > 1:
            if self.out_type == 'mix':
                processed_mags = torch.sum(processed_mags, axis=1)
        else:
            processed_mags = processed_mags.squeeze(1)
        return processed_mags, out_dict