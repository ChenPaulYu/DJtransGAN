import torch
import copy
import torch.nn as nn

from djtransgan.model import Encoder, Frontend, PostProcessor, Unzipper
from djtransgan.model import estimate_postprocessor_in_dim, estimate_postprocessor_out_dims
from djtransgan.mixer import Mixer

class Generator(nn.Module):
    def __init__(self, 
                 unzipper_args, 
                 encoder_args, 
                 context_args, 
                 processor_args, 
                 mixer_args):
        
        super(Generator, self).__init__()
        
        self.data_types      = ['prev', 'next']
        self.hyper           = self.get_hyperparameter(unzipper_args, 
                                               encoder_args, 
                                               context_args, 
                                               processor_args, 
                                               mixer_args)
        self.frontend        = Frontend()        
        self.unzipper        = Unzipper(**unzipper_args)
        
        processor_args['in_dim']        = estimate_postprocessor_in_dim(encoder_args)
        processor_args['out_dims']      = estimate_postprocessor_out_dims(processor_args, unzipper_args)
        processor_args['last_activate'] = 'sigmoid'
        
        
        self.encoder         = Encoder(encoder_args)
        self.post_processors = nn.ModuleList([PostProcessor(processor_args, context_args) for i in range(2)])
        self.mixers          = [Mixer(mixer_args.get('band_args'), 
                                      {**mixer_args.get('fader_args'), 'fade_type': fade_type}) for fade_type in ['fo', 'fi']]
        
        
    def get_hyperparameter(self, 
                           unzipper_args, 
                           encoder_args, 
                           context_args, 
                           processor_args, 
                           mixer_args):
        return {
            'unzipper_args' : copy.deepcopy(unzipper_args), 
            'encoder_args'  : copy.deepcopy(encoder_args),
            'context_args'  : copy.deepcopy(context_args), 
            'processor_args': copy.deepcopy(processor_args), 
            'mixer_args'    : copy.deepcopy(mixer_args)
        }
    
        
    def encode(self, in_waves, phase=False):
        out_tuple  = ()
        encodeds   = [self.frontend(in_wave, phase=phase) for in_wave in in_waves]
        mel_mags    = [encoded[0] for encoded in encodeds]
        mags       = [encoded[1] for encoded in encodeds]
        vecs       = [self.encoder(mel_mag) for mel_mag in mel_mags]
        out_tuple  = (vecs, mags)
        
        if phase:
            phases     = [encoded[2] for encoded in encodeds]
            out_tuple += (phases,) 
        return out_tuple
    
    def render(self, in_mags, in_vecs, cue_region=None):
        render_params  = [self.unzipper(processor(*in_vecs)) for processor in self.post_processors]
        render_outs    = [mixer(in_mag, 
                                band_params  = render_param.get('band'), 
                                fader_params = render_param.get('fader'), 
                                cue_region   = cue_region) for (in_mag, render_param, mixer) in zip(in_mags, render_params, self.mixers)]
        return render_outs
    
    def mix(self, in_datas):
        return torch.sum(torch.stack(in_datas, axis=1), axis=1)

    
    # Apply Inverse STFT to get audio back
    def infer(self, 
              *in_waves, 
              cue_region=None, 
              mix=True):
        
        with torch.no_grad():

            length         = in_waves[0].size(-1)
            in_vecs, in_mags, in_phases = self.encode(in_waves, phase=True)
            
            render_outs    =  self.render(in_mags, in_vecs, cue_region=cue_region)
            out_results    = {k: render_out[1] for (k, render_out) in zip(self.data_types, render_outs)}
            
            out_mags       = [render_out[0] for render_out in render_outs] 
            out_mags       = [torch.sum(out_mag, axis=1) if len(out_mag.size()) == 5 else out_mag for out_mag in out_mags]   
            out_waves      = [self.frontend.inverse(mag, phase, length) for (mag, phase) in zip(out_mags, in_phases)]
            
            if mix:
                out_datas  = self.mix(out_waves)
            else:
                out_datas  = {key: wave for (key, wave) in zip(self.data_types, out_waves)}
                
            return out_datas, out_results

        
    def forward(self, 
                *in_waves, 
                cue_region=None, 
                mix=True):
        
        length  = in_waves[0].size(-1)
        encoded = self.encode(in_waves)
        in_vecs, in_mags = encoded

        render_outs =  self.render(in_mags, in_vecs, cue_region=cue_region)
        out_results = {key: render_out[1] for (key, render_out) in zip(self.data_types, render_outs)}
        out_datas   = [render_out[0] for render_out in render_outs] 
        
        if mix:
            out_datas = self.mix(out_datas)
            
            
        return out_datas, out_results
        