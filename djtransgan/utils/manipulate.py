import torch
import pyloudnorm   as pyln
import pyrubberband as pyrb

from djtransgan.config import settings 
from djtransgan.utils  import squeeze_dim


def normalize_loudness(audio, loudness=-12): # unit: db
    meter          = pyln.Meter(settings.SR)
    if isinstance(audio, torch.Tensor):
        audio      = squeeze_dim(audio).numpy()
        measured   = meter.integrated_loudness(audio)
        normalized = pyln.normalize.loudness(audio, measured, loudness)
        normalized = torch.from_numpy(normalized).unsqueeze(0)
    else:
        measured   = meter.integrated_loudness(audio)
        normalized = pyln.normalize.loudness(audio, measured, loudness)
    return normalized

def normalize_peak(audio, peak_loudness=-1):
    if isinstance(audio, torch.Tensor):
        audio      = squeeze_dim(audio).numpy()
        normalized = pyln.normalize.peak(audio, -1.0)
        normalized = torch.from_numpy(normalized).unsqueeze(0)
    else:
        normalized = pyln.normalize.peak(audio, -1.0)
    return normalized

def normalize(audio, norm_type='loudness'):
    if norm_type is None:
        return audio
    norm_dict = {
        'peak'    : normalize_peak(audio),
        'loudness': normalize_loudness(audio), 
    }
    normalized = norm_dict.get(norm_type, None)
    return audio if normalized is None else normalized


def get_stretch_ratio(src, tgt):
    return tgt / src

def time_stretch(audio, ratio, sr=settings.SR):
    if ratio == 1:
        return audio
    if isinstance(audio, torch.Tensor): 
        stretched = torch.from_numpy(pyrb.time_stretch(squeeze_dim(audio).numpy(), sr, ratio)).unsqueeze(0)
    else:
        stretched = pyrb.time_stretch(audio, sr, ratio)
    return stretched
