import torch

from djtransgan.config  import settings
from djtransgan.utils   import load_audio
from djtransgan.utils   import get_audio_info, pad_tensor, time_to_samples, samples_to_time

    
    
def get_new_cue(cue, cue_mid, step, ratio):
    new_cue = [c-(cue_mid-step) for c in cue]
    if ratio:
        new_cue = [c / (2*step) for c in new_cue]
    else:
        new_cue = [samples_to_time(c) for c in new_cue]
    return torch.tensor(new_cue).float()



def select_audio_region(in_data, 
                        cue, 
                        n_time, 
                        ratio, 
                        select_idx):
    
    data_type  = 'tensor' if isinstance(in_data, torch.Tensor) else 'str'
    length     = in_data.size(-1) if data_type == 'tensor' else get_audio_info(in_data).num_frames
    cue        = [time_to_samples(c) for c in cue] 
    step       = int(time_to_samples(n_time/2))
    cue_mid    = int(sum(cue) // 2)

    time_dict  = [
        [0 if cue_mid-step < 0 else cue_mid-step, cue[1]], 
        [cue[0], length if cue_mid+step > length else cue_mid+step],
        [0 if cue_mid-step < 0 else cue_mid-step, length if cue_mid+step > length else cue_mid+step]
    ]
    timestamps = time_dict[select_idx]
    
    if data_type == 'tensor':
        audio = in_data[:, timestamps[0]:timestamps[1]]
    else:
        audio = load_audio(in_data, 
                           start = timestamps[0], 
                           end   = timestamps[1])
    
    pad = cue_mid + step - length 
    if pad > 0:
        audio  = pad_tensor(audio,
                           pad, 
                           value=0., 
                           last=True)
        
    pad = 2*step - audio.size(-1)
    if pad > 0:
        audio = pad_tensor(audio, 
                           pad, 
                           value=0., 
                           last=(1-select_idx)>0)
    new_cue   = get_new_cue(cue, cue_mid, step, ratio)
    return audio, new_cue, (cue, timestamps)