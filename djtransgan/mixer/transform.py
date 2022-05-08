import math
import torch


def choose_fade_shape(fade, fade_shape='linear'):
    transforms = {
        'linear'      : fade, 
        'exponential' : torch.pow(2, (fade - 1)) * fade, 
        'logarithmic' : (torch.log10(.1 + fade) + 1) / 1.0414, 
        'quarter_sine': torch.sin(fade * math.pi / 2), 
        'half_sine'   : torch.sin(fade * math.pi - math.pi / 2) / 2 + 0.5
    }
    if fade_shape not in transforms.keys(): return
    return transforms[fade_shape]

def fade_transform(curve, fade_type, fade_shape):
    return {
        'fi': choose_fade_shape(curve, fade_shape), 
        'fo': choose_fade_shape(1. - curve, fade_shape)
    }[fade_type]

def linear_transform(v, minv, maxv):  
    return (maxv - minv) * v + minv

def linear_transform_inv(v, minv, maxv, maxy=6):
    return maxy / (maxv - minv) / v
