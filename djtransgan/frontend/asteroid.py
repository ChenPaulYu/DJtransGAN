import torch
import torch.nn as nn
from openunmix.transforms import make_filterbanks 
from djtransgan.config    import settings 

class AsteroidSTFT(nn.Module):
    def __init__(self, 
                 n_fft=settings.N_FFT, 
                 hop_length=settings.HOP_LENGTH, 
                 sr=settings.SR, 
                 center=True,
                 power=1, 
                 length=None):

        super(AsteroidSTFT, self).__init__()
        self.power   = power
        fbs          = make_filterbanks(n_fft=n_fft, 
                                        n_hop=hop_length, 
                                        center=center, 
                                        sample_rate=sr, 
                                        method='asteroid')
        self.length  = length 
        self.encoder = fbs[0]
        self.decoder = fbs[1]
        
        
    def inverse(self, mags, phases, length=None):
        if self.power == 2:
            mags = torch.sqrt(mags)
        reals    = mags * torch.cos(phases) 
        imags    = mags * torch.sin(phases)   
        matrixs  = torch.stack((reals, imags), axis=-1)
        length   = length if length is not None else self.length
        return self.decoder(matrixs, length=length)
        
        
    def forward(self, waves, phase=True):
        matrixs = self.encoder(waves)
        reals   = matrixs[..., 0]
        imags   = matrixs[..., 1]
        mags    = reals**2 + imags**2
        if self.power == 1:
            mags = torch.sqrt(mags)
        
        if phase:
            phases = torch.atan2(imags, reals)        
            return mags, phases
        return mags


