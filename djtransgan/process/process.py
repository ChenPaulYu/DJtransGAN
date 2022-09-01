import torch

from djtransgan.config  import settings
from djtransgan.utils   import normalize, load_audio, samples_to_time, squeeze_dim
from djtransgan.dataset import select_audio_region
from djtransgan.process import sync_bpm, sync_cue
from djtransgan.process import estimate_beat, select_cue_points, correct_cue


def preprocess(prev_audio, 
               next_audio, 
               prev_cue, 
               next_cue):
    
    
    
    print('[1/5] beat tracking start ...')
    _, prev_bpm, _, prev_downbeat = estimate_beat(prev_audio)
    _, next_bpm, _, next_downbeat = estimate_beat(next_audio)
    print('[1/5] beat tracking complete ...')
    
    print('[2/5] bpm matching start ...')
    next_audio, ratio = sync_bpm(next_audio, prev_bpm, next_bpm)
    next_downbeat     = next_downbeat/ratio
    next_cue          = correct_cue(next_downbeat, next_cue/ratio)
    prev_cue          = correct_cue(prev_downbeat, prev_cue)
    print('[2/5] bpm matching complete ...')
    
    print('[3/5] cue point select start ...')
    prev_cues, next_cues  = select_cue_points(prev_cue, next_cue, prev_downbeat, next_downbeat)
    print('[3/5] cue point select complete ...')
    
    print('[4/5] cue region alignment start ...')
    next_audio, next_cues = sync_cue(prev_audio, next_audio, prev_cues, next_cues)
    print('[4/5] cue region alignment complete ...')
        
    print('[5/5] normalize start ...')
    next_audio    = normalize(next_audio)
    prev_audio    = normalize(prev_audio)
    print('[5/5] normalize complete ...')
        
    
    pair_audio_for_g = []
    cue_for_g        = []
    
    prev_audio_for_g, prev_cues_for_g, (prev_cues_ori, prev_timestamps) = select_audio_region(prev_audio, 
                                                                                              prev_cues,
                                                                                              settings.N_TIME, 
                                                                                              True, 
                                                                                              0)
    next_audio_for_g, next_cues_for_g, (next_cues_ori, next_timestamps) = select_audio_region(next_audio, 
                                                                                              next_cues,
                                                                                              settings.N_TIME, 
                                                                                              True, 
                                                                                              1)
    pair_audio       = [prev_audio, next_audio]
    cue_ori          = [prev_cues_ori, next_cues_ori]
    timestamps       = [prev_timestamps , next_timestamps]
    pair_audio_for_g = [prev_audio_for_g.unsqueeze(0), next_audio_for_g.unsqueeze(0).to(torch.float32)]
    cue_for_g        = prev_cues_for_g.unsqueeze(0).to(torch.float32)
    
    return (pair_audio, timestamps), (pair_audio_for_g, cue_for_g)
    
    
    
    
def postprocess(mix_audio, pair_audio, timestamp, cue):
        
    
    if len(mix_audio.size()) > 2:
        mix_audio = squeeze_dim(mix_audio)
    
    if len(cue.size()) > 1:
        cue = squeeze_dim(cue)
    
    new_cue    = samples_to_time(timestamp[0][0] + cue * settings.N_TIME * settings.SR)
    
    prev_audio = pair_audio[0][:, :timestamp[0][0]]
    next_audio = pair_audio[1][:, timestamp[1][1]:]
    
    
    
    
    return torch.cat((prev_audio, mix_audio, next_audio), axis=1), new_cue
    