import os
import math 
import torch
import torchaudio
import numpy as np
import IPython.display   as ipd
import matplotlib.pyplot as plt

from scipy.fftpack          import fft
from matplotlib.offsetbox   import AnchoredText

from djtransgan.config      import settings
from djtransgan.utils.utils import squeeze_dim

    
def wave_visualize(waveform, plt, title=None, timestamps=None, **kargs):
    if isinstance(waveform, torch.Tensor): 
        waveform   = squeeze_dim(waveform).numpy()
    if isinstance(timestamps, torch.Tensor): 
        timestamps = timestamps.numpy()
        
    N = len(waveform)
    T = len(waveform)/settings.SR
    x = np.linspace(0.0, T, N)
    y = waveform
    
    plt.plot(x, y, **kargs)
    if timestamps is not None:
        for timestamp in timestamps:
            plt.axvline(x=timestamp, c='red', ls='dashed', lw=0.5)
    if title: 
        plt.set_title(title)
        
        
def audio_visualize(waveform): 
    if isinstance(waveform, torch.Tensor): waveform=squeeze_dim(waveform).numpy()
    ipd.display(ipd.Audio(waveform, rate=settings.SR))

    
def fft_visualize(waveform, plt, title=None, sr=settings.SR):
    if isinstance(waveform, torch.Tensor): 
        waveform   = squeeze_dim(waveform).numpy()
    T       = 1.0 / sr
    N       = waveform.shape[0]
    yf      = fft(waveform)
    yf      = 2.0 / N * np.abs(yf[:N//2])
    xf      = np.linspace(0.0, 1.0/(2.0*T), N//2)
    
    max_index  = np.argmax(yf)
    cut_cands  = np.where(yf > (np.max(yf) / 20))
    cut_index  = np.max(cut_cands) if len(cut_cands[0]) != 0 else max_index * 2
    cut_index += (len(yf) - cut_index) // 100     
    plt.plot(xf[:cut_index], yf[:cut_index])
    
    if title: 
        plt.set_title(title)
        
    try:
        anchor_text1 = AnchoredText(f'max f: {np.round(xf[max_index] , 2)} hz', loc=2)
        anchor_text2 = AnchoredText(f'cut f: {np.round(xf[cut_index] , 2)} hz', loc=1)
        plt.add_artist(anchor_text1)
        plt.add_artist(anchor_text2)
    except:
        pass
    
def mask_visualize(mask, plt, title=None, cmap=None):
    if isinstance(mask, torch.Tensor): 
        mask   = squeeze_dim(mask)

    x = torch.arange(mask.size(1))
    y = torch.arange(mask.size(0))
    if cmap is None:
        im = plt.pcolormesh(x, y, mask)
    else:
        im = plt.pcolormesh(x, y, mask, cmap=cmap)
        
    if title:
        plt.set_title(title)
        
    plt.axis('off')
    
def loss_visualize(loss, plt, title=None):
    
    if isinstance(loss, list): 
        loss = torch.tensor(loss)
        
    plt.plot(loss['G'], label='G Loss')
    plt.plot(loss['D'], label='D Loss')
    
    if title:   
        plt.suptitle(title)
    
    plt.legend(loc='best')
    
    return plt
        
        
def save_figure(plt, out_path):
    plt.savefig(out_path, transparent=False, facecolor='w', edgecolor='w', orientation='portrait')