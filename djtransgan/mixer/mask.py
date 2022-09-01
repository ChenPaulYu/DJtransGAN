import torch
import torch.nn as nn

from djtransgan.config      import settings
from djtransgan.mixer.fader import Relu6Faders
from djtransgan.utils.utils import purify_device


BAND_TO_FADE = {
    'low' : 'fo', 
    'high': 'fi'
}




class VMask(nn.Module):
    def __init__(self, sum_type, band_type, fade_shape, split_type):
        
        '''
        Args:
            sum_type  :  (mean, sum)
            band_type :  (low, high)
            fade_shape:  (linear, exponential, logarithmic, quarter_sine, half_sine)
            split_type:  (bias, equal)
            
        '''
        
        super(VMask, self).__init__()
        self.nyqf       = settings.SR / 2
        self.band_freqs = settings.BAND_FREQS
        self.fader      = Relu6Faders(sum_type, BAND_TO_FADE[band_type], fade_shape)
        self.band_type  = band_type
        self.split_type = split_type
        
    def freq2ratio(self, freq):
        return freq / self.nyqf
        
    def get_bounded_ratio(self, index, n_band):
        if self.split_type == 'bias':
            min_freq = self.band_freqs[index]
            max_freq = self.band_freqs[index + 1]
            return self.freq2ratio(min_freq), self.freq2ratio(max_freq)
        elif self.split_type == 'equal':
            unit     = self.nyqf / n_band
            min_freq = unit * index
            max_freq = unit * (index + 1)    
            return self.freq2ratio(min_freq), self.freq2ratio(max_freq)
        
    def get_bounded_ratios(self, params, total_band, start_index):
        
        if total_band is None:
            n_band  = params.size(1) + 1
        else:
            n_band  = total_band
        bounded_ratios = [self.get_bounded_ratio(index + start_index, n_band) for index in range(params.size(1))]
        return bounded_ratios
    
    def curve_fillout(self, curves):
        n_batch, _, n_channel, n_bins = curves.size()
        zeros = torch.zeros(n_batch, 1, n_channel, n_bins, device=curves.device)
        ones  = torch.ones(n_batch , 1, n_channel, n_bins, device=curves.device)   
        
        return {
            'low' : torch.cat((zeros, curves, ones), axis=1),
            'high': torch.cat((zeros, torch.flip(curves, dims=[1]), ones), axis=1)
        }[self.band_type]
        
    
    def band_exciter(self, curves):
        curves_fill = self.curve_fillout(curves)
        curves_diff = torch.diff(curves_fill, axis=1)
        return curves_diff
    
    def create_fake_waves(self, mags):
        return torch.ones_like(mags[..., 0], device=mags.device)
        
    def create_mask(self, curves, n_frames):
        expand_dims = list(curves.size()) + [n_frames]
        return curves.unsqueeze(-1).expand(expand_dims)
    
    def masking(self, mags, curves):
        masks = self.create_mask(curves, mags.size(-1))
        mags  = mags.unsqueeze(1).expand_as(masks)
        return mags * masks, masks
        
        
    
    def forward(self, mags, params, total_band=None, start_index=0):
        '''
        Args:
            mags (Tensor)  : (n_batch, n_channel , n_bins, n_frames)
            params (Tensor): (n_batch, n_band - 1, 2)
            
        Return:

        '''
        n_batch, n_channel , n_bins, n_frames = mags.size()
        fake_waves     = self.create_fake_waves(mags)
        bounded_ratios = self.get_bounded_ratios(params, total_band, start_index)
        
        curves         = torch.stack([self.fader(fake_waves, 
                                                 params[:, index:index+1, :], 
                                                 bounded_ratios[index][0], 
                                                 bounded_ratios[index][1]) [0] for index in range(params.size(1))], axis=1)
        processed_curves                = self.band_exciter(curves)
        processed_mags, processed_masks = self.masking(mags, processed_curves)
        return processed_mags, purify_device(processed_masks), purify_device(processed_curves), purify_device(curves)
    
    
    
    
class HMask(nn.Module):
    def __init__(self, sum_type, fade_type, fade_shape):
        
        '''
        Args:
            sum_type  :  (mean, sum)
            fade_type :  (fi, fo)
            fade_shape:  (linear, exponential, logarithmic, quarter_sine, half_sine)            
        '''
        
        super(HMask, self).__init__()
        self.nyqf       = settings.SR / 2
        self.fader      = Relu6Faders(sum_type, fade_type, fade_shape)
        self.fade_type  = fade_type
        
        
        
    def create_fake_waves(self, mags):
        return torch.ones_like(mags[..., 0, :], device=mags.device)
    
    def create_mask(self, curves, n_bins):
        expand_dims = list(curves.size())
        expand_dims.insert(-1, n_bins)
        return curves.unsqueeze(-2).expand(expand_dims)
    
    def masking(self, mags, curves):
        masks = self.create_mask(curves, mags.size(-2))
        return mags * masks, masks
        
        
    def forward(self, mags, params, cue_region=None):
        '''
        Args:
            mags (Tensor)  : (n_batch, n_channel, n_bins, n_frames)
            params (Tensor): (n_batch, n_fader, n_params)
            
        Return:

        '''
        n_batch        = params.size(0)
        n_band         = params.size(1)
        cue_region       = torch.tensor([0, 1]).expand(n_batch, -1) if cue_region is None else cue_region
        fake_waves       = self.create_fake_waves(mags)
        processed_curves = self.fader(fake_waves, params, cue_region[..., 0], cue_region[..., 1])[0]
        processed_mags, processed_masks = self.masking(mags, processed_curves)
        return processed_mags, purify_device(processed_masks), purify_device(processed_curves)
    