import torch
import torch.nn as nn
import torchaudio

from djtransgan.config      import settings
from torchaudio.transforms  import AmplitudeToDB, MelScale, Spectrogram 
from djtransgan.frontend    import AsteroidSTFT
from djtransgan.frontend    import TorchlibrosaSTFT
from djtransgan.frontend    import NNaudioSTFT

def get_amp2db_func(power=1):
    return AmplitudeToDB(stype=['amplitude', 'power'][int(power)-1])


def get_stft_func(**kargs): # stype: asteroid, nnaudio, torchlibrosa
    
    stft_type = kargs.get('stft_type')
    if stft_type:
        kargs.pop('stft_type')
    else:
        stft_type = 'torchlibrosa'
    
    return {
        'nnaudio' : NNaudioSTFT(**kargs),
        'asteroid': AsteroidSTFT(**kargs), 
        'torchlibrosa': TorchlibrosaSTFT(**kargs), 
    }[stft_type]


def get_mel_func():
    return MelScale(n_mels=settings.N_MELS, 
                    n_stft=settings.N_FFT // 2 + 1, 
                    sample_rate=settings.SR)