import re
import os
import math
import json
import torch
import random
import joblib
import librosa
import torchaudio
import numpy as np
import torch.nn.functional as F

from djtransgan.config import settings

random.seed(settings.RANDOM_SEED)
torchaudio.set_audio_backend('soundfile')

# Transform

def to_mono(audio, dim=-2): 
    if len(audio.size()) > 1:
        return torch.mean(audio, dim=dim, keepdim=True)
    else:
        return audio
    
def time_to_samples(time): 
    return librosa.time_to_samples(time, sr=settings.SR)

def samples_to_time(samples):
    return librosa.samples_to_time(samples, sr=settings.SR)

def time_to_str(secs):
    return f'{int(secs/60)}:{(secs%60):.2f}'
    
def str_to_time(string):
    mins, secs = string.split(':')
    return mins*60+secs
    

# I/O

def load_audio(audio_path, 
               sr=settings.SR, 
               mono=True, 
               start=0, 
               end=None
              ):
    if 'mp3' in audio_path:
        torchaudio.set_audio_backend('sox_io')
    audio, org_sr = torchaudio.load(audio_path, frame_offset=start, num_frames=-1 if end is None else end-start)
    audio = to_mono(audio) if mono else audio
    
    if org_sr != sr:
        audio = torchaudio.transforms.Resample(org_sr, sr)(audio)

    return audio

def out_audio(data, out_path, sr=settings.SR):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    if len(data.size()) == 1: data = data.unsqueeze(0)
    torchaudio.save(out_path, data, sr)

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def out_json(data, out_path, cover=True):
    if check_exist(out_path) and cover == False: return
    with open(out_path, 'w') as outfile: json.dump(data, outfile)
        
    
def load_npy(in_path):
    return np.load(in_path, allow_pickle=True)
        
def out_npy(data, out_path):
    check_exist(out_path)
    np.save(out_path.split('.npy')[0], data)

def load_pt(in_path):
    return torch.load(in_path)
    

def out_pt(data, out_path):
    check_exist(out_path)
    torch.save(data, out_path)
    

    
# Others
    
def check_exist(out_path):    
    if re.compile(r'^.*\.[^\\]+$').search(out_path):
        out_path = os.path.split(out_path)[0]        
    existed = os.path.exists(out_path)
    if not existed:
        os.makedirs(out_path, exist_ok=True)
    return existed

def check_extent(file, exts):
    if isinstance(exts, str):
        return file.endswith(f'.{exts}')
    else:
        return bool(sum([file.endswith(f'.{ext}') for ext in exts]) > 0)
    
def get_list_intersect(lst1, lst2):
    return list(set(lst1) & set(lst2))

def get_extention(data_path):
    return os.path.splitext(audio_path)[-1][1:]

def get_filename(data_path):
    return os.path.splitext(data_path)[0].split('/')[-1]

def get_class_name(class_):
    return class_.__class__.__name__

def random_samples(arr, n_sample):
    return arr if len(arr) < n_sample else random.sample(arr, n_sample)
    
def pad_tensor(x, num_samples, value=None, last=True):
    if value is None:
        pads = (torch.rand(num_samples) * 2 - 1).unsqueeze(0) * 0.1
    elif value == -1:
        if last:
            pads = torch.linspace(1, settings.EPSILON, num_samples).unsqueeze(0) * 0.1
        else:
            pads = torch.linspace(0, settings.EPSILON, num_samples).unsqueeze(0) * 0.1
        pads *= 0.1
    elif value == 0.:
        pads = torch.full((x.size(-2), num_samples), settings.EPSILON)    
    else:
        pads = torch.full((x.size(-2), num_samples), value)   
    return [torch.cat((pads, x), axis=-1), torch.cat((x, pads), axis=-1)][last]

    
def squeeze_dim(data):
    dims = [i for i in range(len(data.size())) if data.size(i) == 1]
    for dim in dims:
        data = data.squeeze(dim)
    return data

def unsqueeze_dim(data, num):
    for i in range(num):
        data = data.unsqueeze(-1)
    return data

def get_device(n_gpu):
    return torch.device('cpu' if int(n_gpu) == -1 else f'cuda:{n_gpu}')

def purify_device(data, idxs=None):    
    if idxs is None: 
        idxs = range(len(data))
    if isinstance(data, list):        
        return [d.detach().cpu().clone() if i in idxs else d for i, d in enumerate(data)]
    if isinstance(data, tuple):
        return tuple([d.detach().cpu().clone() if i in idxs else d for i, d in enumerate(data)])
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().clone()
    
def reduce(x, reduction=None):
    if reduction == "mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:
        return x
    
def find_index(arr, val):
    return np.where(arr == val)[0][0]

def find_nearest(arr, val):
    return np.argmin(abs(arr - val))
    
def check_nan(data):
    return (torch.sum(torch.isnan(torch.tensor(data))) >= 1).item()

def get_audio_info(audio_path):
    if 'mp3' in audio_path:
        torchaudio.set_audio_backend('sox_io')
    return torchaudio.info(audio_path)