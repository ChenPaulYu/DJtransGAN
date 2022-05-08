import torch
import torch.nn as nn
from nnAudio import Spectrogram
from djtransgan.config import settings 

class NNaudioSTFT(torch.nn.Module):
    def __init__(self, 
                 n_fft=settings.N_FFT, 
                 hop_length=settings.HOP_LENGTH, 
                 sr=settings.SR, 
                 power=1, 
                 center=True, 
                 length=None):
        super(NNaudioSTFT, self).__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.power      = power
        self.center     = center
        self.length     = length
        self.stft       = Spectrogram.STFT(n_fft=n_fft, 
                                           hop_length=512,
                                           center=center,
                                           sr=sr, 
                                           iSTFT=True)
        
        
    def inverse(self, mags, phases, length=None):
        
        if self.power == 2:
            mags = torch.sqrt(mags)
        
        matrixs     = torch.stack([mags*torch.cos(phases), 
                               mags*torch.sin(phases)], dim=-1)
        stft        = self.stft.to(mags.device)
        length      = length if length is not None else self.length
        if length is not None:
            waves   = stft.inverse(matrixs, onesided=True, length=length)
        else:
            waves   = stft.inverse(matrixs, onesided=True)
        return waves.unsqueeze(1)

        
    def forward(self, waves, phase=True):
        stft   = self.stft.to(waves.device)
        specs  = stft(waves)
        reals  = specs[..., 0]
        imags  = specs[..., 1]
        mags   = torch.sqrt(reals**2 + imags**2)
        if phase:
            phases = torch.atan2(imags, reals)        
            return mags, phases
        return mags,