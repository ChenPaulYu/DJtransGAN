import torch

from djtransgan.config  import settings
from djtransgan.utils   import load_audio, time_to_samples
from djtransgan.utils   import get_stretch_ratio, time_stretch
from djtransgan.process import split_audio



def sync_cue(prev_audio, next_audio, prev_cue, next_cue):   
    prev_sample = prev_audio[..., time_to_samples(prev_cue[0]):time_to_samples(prev_cue[1])].size(-1)
    next_sample = next_audio[..., time_to_samples(next_cue[0]):time_to_samples(next_cue[1])].size(-1)
    
    # time inverse inversely proportional with bpm
    ratio         = get_stretch_ratio(next_sample, prev_sample)
    splited       = split_audio(next_audio, next_cue)
    splited[1]    = time_stretch(splited[1], ratio)
    begin_idx     = 1 if splited[0] is None else 0
    synced        = splited[begin_idx]
    next_cue      = [next_cue[0], 
                     next_cue[0]+(prev_cue[1]-prev_cue[0])]
    
    for s in splited[begin_idx+1:]:
        synced    = torch.cat((synced, s), 1)
        
    return  synced, next_cue

def sync_bpm(next_audio, 
             prev_bpm, 
             next_bpm):
    ratio      = get_stretch_ratio(next_bpm, prev_bpm)
    next_audio = time_stretch(next_audio, ratio) 
    return next_audio, ratio