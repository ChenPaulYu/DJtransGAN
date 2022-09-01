import numpy as np
from djtransgan.config import settings 
from djtransgan.utils  import find_nearest, find_index, time_to_samples


def select_cue_points(prev_cue, 
                      next_cue, 
                      prev_downbeat, 
                      next_downbeat):
    prev_cues     = [prev_downbeat[find_index(prev_downbeat, prev_cue)-settings.CUE_BAR], prev_cue]
    next_cues     = [next_downbeat[find_index(next_downbeat, next_cue)-settings.CUE_BAR], next_cue]
    return prev_cues, next_cues
    

def correct_cue(downbeat, cue):
    return downbeat[find_nearest(downbeat, cue)] 


def filter_beat(beat, downbeat, sig_d):
    begin = np.where(beat == downbeat[0])[0][0]
    end   = np.where(beat == downbeat[-1])[0][0] + sig_d
    return beat[begin:end]


def split_audio(audio, cue):
    splited      = ()
    cue_sample   = [time_to_samples(c) for c in cue]
    before       = None if cue_sample[0] == 0 else  audio[:, :cue_sample[0]]
    middle       = audio[:, cue_sample[0]:cue_sample[1]]
    after        = None if cue_sample[1] == 0 else audio[:, cue_sample[1]:]
    return [before, middle, after]