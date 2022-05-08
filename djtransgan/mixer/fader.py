import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd  import Function

from djtransgan.utils.utils      import purify_device, unsqueeze_dim
from djtransgan.mixer.transform  import linear_transform, linear_transform_inv, fade_transform


class custom_relu6(Function):
    
    @staticmethod
    def forward(self, inp): 
        return F.relu6(inp)
    
    @staticmethod
    def backward(self, grad_out): 
        return grad_out
    
    
def prelu6(x, start, sacle, mode='custom'):
    if mode == 'custom':
        y = custom_relu6.apply(custom_relu6.apply(x - start) * sacle)
    else:
        y = F.relu6(F.relu6(x - start) * sacle)
    return y


class Relu6Faders(nn.Module):
    
    def __init__(self, sum_type, fade_type, fade_shape):
        super(Relu6Faders, self).__init__()
        self.sum_type   = sum_type
        self.fade_type  = fade_type
        self.fade_shape = fade_shape
    
    def calibrate(self, curves):
        return curves - curves[..., :1]
    
    def normalize(self, curves):
        return curves / curves[..., -1:]
    
    def expand_to(self, source, target):
        return source.unsqueeze(-1).expand_as(target)
    
    def unzip(self, params, minv, maxv):
        if self.sum_type == 'mean':
            if isinstance(minv, torch.Tensor): 
                minv = unsqueeze_dim(minv, len(params[..., 0].size()) - len(minv.size()))
            if isinstance(maxv, torch.Tensor): 
                maxv = unsqueeze_dim(maxv, len(params[..., 0].size()) - len(maxv.size()))
            
            start = linear_transform(params[..., 0], minv, maxv)
            slope = linear_transform_inv(params[..., 1], start, maxv)
            if params.size(-1) > 2: 
                weight = params[..., 2]
            else:
                return [start, slope]
        elif self.sum_type == 'sum':
            new_params = [[], []]
            if isinstance(minv, torch.Tensor): 
                minv = unsqueeze_dim(minv, len(params[..., 0, 0].size()) - len(minv.size()))
            if isinstance(maxv, torch.Tensor): 
                maxv = unsqueeze_dim(maxv, len(params[..., 0, 0].size()) - len(maxv.size()))
            
            for index in range(params.size(-2)):
                start  = linear_transform(params[..., index, 0], minv , maxv)
                slope  = linear_transform_inv(params[..., index, 1], start, maxv)
                new_params[0].append(start)
                new_params[1].append(slope)
                if params.size(-1) > 2: 
                    weight = params[..., 2]
                    new_params[2].append(weight)
                minv   = start + (maxv / slope)
            return [torch.stack(new_param, axis=-1) for new_param in new_params]
        
    
    def get_raw_curve(self, waves, params):
        x = torch.linspace(0, 6, waves.size(-1), device=waves.device)
        if len(params.size()) > 3:
            num_batch, num_wave, num_fader, num_param = params.size()
            return x.expand(num_batch, num_wave, num_fader, -1)
        else:
            num_batch, num_fader, num_param = params.size()
            return x.expand(num_batch, num_fader, -1)
    
    def render_curve(self, x, start, slope):
        curves = prelu6(x, start, slope) / 6
        curves = fade_transform(curves, self.fade_type, self.fade_shape)
        return curves
    
    def sum_curves(self, curves, waves):
        if self.sum_type == 'mean':
            sum_curve = torch.mean(curves, dim=-2)
            sum_curve = sum_curve.unsqueeze(-2).expand_as(waves) 
            return sum_curve
        elif self.sum_type == 'sum':
            sum_curve = torch.sum(curves, axis=-2)
            sum_curve = self.normalize(self.calibrate(sum_curve))
            sum_curve = sum_curve.unsqueeze(-2).expand_as(waves) 
            return sum_curve
            
        
    def forward(self, waves, params, minv=0, maxv=1):

        '''
        Args:
            wave (Tensor)  : (num_batch, num_channel, num_samples)
            params (Tensor): (num_batch, num_fader  , num_param)
            
            or
            
            wave (Tensor)  : (num_batch, num_wave, num_fader, num_samples)
            params (Tensor): (num_batch, num_wave, num_fader, num_samples)

        '''
        params           = params[:waves.size(0), ...]
        raw_curves       = self.get_raw_curve(waves, params)
        unzip_params     = self.unzip(params, minv * 6, maxv * 6)
        params           = [self.expand_to(param, raw_curves) for param in unzip_params]
        processed_curves = self.render_curve(raw_curves, params[0], params[1])
        processed_curves = processed_curves * params[2] if len(params) > 2 else processed_curves
        processed_curves = self.sum_curves(processed_curves, waves)
        processed_waves  = processed_curves * waves
        return processed_waves, purify_device(processed_curves)
    
    
