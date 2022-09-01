import torch
import torch.nn as nn
import torch.nn.functional as F
import torchlibrosa as tl
from djtransgan.config import settings 

class TorchlibrosaSTFT(nn.Module):
    def __init__(self, 
                 n_fft=settings.N_FFT, 
                 hop_length=settings.HOP_LENGTH, 
                 sr=settings.SR, 
                 power=1, 
                 length=None):

        super(TorchlibrosaSTFT, self).__init__()
        self.power   = power
        self.length  = length
        self.encoder = tl.STFT(n_fft=n_fft, hop_length=hop_length)
        self.decoder = tl.ISTFT(n_fft=n_fft, hop_length=hop_length)
        
    def swapaxes(self, data):
        return torch.swapaxes(data, 2, 3)
        
        
    def inverse(self, mags, phases, length=None):
        if self.power == 2:
            mags = torch.sqrt(mags)
        reals    = self.swapaxes(mags * torch.cos(phases)) 
        imags    = self.swapaxes(mags * torch.sin(phases)) 
        length   = length if length is not None else self.length
        return self.decoder(reals, imags, length).unsqueeze(1)
        
        
    def forward(self, waves, phase=True):
        reals, imags  = self.encoder(waves.squeeze(1))
        reals  = self.swapaxes(reals)
        imags  = self.swapaxes(imags)
        mags   = reals**2 + imags**2
        if self.power == 1:
            mags = torch.sqrt(mags)
        
        if phase:
            phases = torch.atan2(imags, reals)        
            return mags, phases
        return mags,


    
    