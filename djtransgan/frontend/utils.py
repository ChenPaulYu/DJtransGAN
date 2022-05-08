import torch
import torch.nn as nn
import torchaudio

from djtransgan.config                 import settings
from torchaudio.transforms             import AmplitudeToDB, MelScale, Spectrogram 
from djtransgan.frontend.asteroid      import AsteroidSTFT
from djtransgan.frontend.torchlibrosa  import TorchlibrosaSTFT
from djtransgan.frontend.nnaudio       import NNaudioSTFT

def get_amp2db_func(power=1):
    stype = ['amplitude', 'power'][int(power)-1]
    return AmplitudeToDB(stype=stype)


def get_stft_func(**kargs): # stype: asteroid, nnaudio, torchlibrosa
    
    stype = kargs.get('stype')
    if stype:
        kargs.pop('stype')
    else:
        stype = 'torchlibrosa'
    
    return {
        'nnaudio' : NNaudioSTFT(**kargs),
        'asteroid': AsteroidSTFT(**kargs), 
        'torchlibrosa': TorchlibrosaSTFT(**kargs), 
    }[stype]


def get_mel_func():
    sr     = settings.SR
    n_mels = settings.N_MELS
    n_stft = settings.N_FFT // 2 + 1
    return MelScale(n_mels=n_mels, n_stft=n_stft, sample_rate=sr)


def gain_normalize(audio):
    max_v = torch.max(audio, axis=-1).values
    min_v = torch.min(audio, axis=-1).values
    
    
    ratio = 2 / (max_v-min_v)
    offset = min_v * ratio + 1
    return audio * ratio - offset