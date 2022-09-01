import os
import torch
from torch.utils.data   import Dataset
from djtransgan.config  import settings
from djtransgan.dataset import generate_noise, select_audio_region
from djtransgan.utils   import load_audio, load_npy, load_json
from djtransgan.utils   import get_list_intersect, time_to_samples, samples_to_time, normalize



class Noise(Dataset):
    def __init__(self, 
                 n_time    = settings.N_TIME, 
                 n_sample  = 100,
                 norm_type = 'loudness'):
        
        self.norm_type = norm_type
        self.n_sample  = n_sample 
        self.noises    = [generate_noise(n_time) for i in range(n_sample)]
    
    def __getitem__(self, index):
        noise = self.noises[index]
        return normalize(self.noises[index].unsqueeze(0), 
                         norm_type=self.norm_type)
        
    def __len__(self):
        return self.n_sample
    
    
class Pair(Dataset):
    
    def __init__(self, 
                 n_time    = settings.N_TIME, 
                 norm_type = 'loudness', 
                 cue_ratio = True):
        
        self.cue_ratio  = cue_ratio
        self.n_time     = n_time
        self.norm_type  = norm_type
        self.audio_dir  = os.path.join(settings.PAIR_DIR, 'audio')
        self.meta_dir   = os.path.join(settings.PAIR_DIR, 'meta')
        self.pair_ids   = self.__get_pair_ids()
        self.data_types = ['prev', 'next'] 
        
    def __get_pair_ids(self):
        audio_ids = list(os.listdir(self.audio_dir))
        meta_ids  = [file.split('.json')[0] for file in os.listdir(self.meta_dir) if file.endswith('.json')]
        return get_list_intersect(audio_ids, meta_ids)
    
    def __get_obj(self, pair_id, data_type):
        
        audio_path = os.path.join(self.audio_dir, pair_id, f'{data_type}.wav')
        meta       = load_json(os.path.join(self.meta_dir  , f'{pair_id}.json')) 
        
        
        return {
            'cue'       : meta[data_type]['cue'], 
            'audio_path': audio_path
        }
    
    def __getitem__(self, index):
        
        pair_id     = self.pair_ids[index]
        pair_objs   = [self.__get_obj(pair_id, data_type) for data_type in self.data_types]
        pair_audios = []
        cues        = []
        sources     = []
        
        for i, pair_obj in enumerate(pair_objs):
            audio, cue, _ = select_audio_region(pair_obj['audio_path'], 
                                                pair_obj['cue'],
                                                self.n_time, 
                                                self.cue_ratio, 
                                                i)
            audio      = normalize(audio, norm_type=self.norm_type)
            
            cues.append(cue)
            pair_audios.append(audio)
        
        return pair_audios, cues[0]
        
        
        
    def __len__(self):
        return len(self.pair_ids)

        
    
class Mix(Dataset):
    def __init__(self,                  
                 n_time    = settings.N_TIME, 
                 norm_type = 'loudness', 
                 cue_ratio = True):
        
        self.cue_ratio  = cue_ratio
        self.n_time     = n_time
        self.norm_type  = norm_type
        self.audio_dir  = os.path.join(settings.MIX_DIR, 'audio')
        self.obj_dir    = os.path.join(settings.MIX_DIR, 'obj')
        self.pair_ids   = self.__get_pair_ids()
        
        
    def __get_pair_ids(self):
        audio_ids = [file.split('.wav')[0] for file in os.listdir(self.audio_dir) if file.endswith('.wav')]
        obj_ids   = [file.split('.npy')[0] for file in os.listdir(self.obj_dir)   if file.endswith('.npy')]
        return get_list_intersect(audio_ids, obj_ids)
    
    def __get_obj(self, pair_id):
        
        audio_path = os.path.join(self.audio_dir, f'{pair_id}.wav')        
        obj        = load_npy(os.path.join(self.obj_dir    , f'{pair_id}.npy')).item()
        
        
        return {
            'cue'       : obj['cue'], 
            'audio_path': audio_path,
        }

    
    def __getitem__(self, index):
        
        pair_id    = self.pair_ids[index]
        pair_obj   = self.__get_obj(pair_id)
        
        audio, cue, _ = select_audio_region(pair_obj['audio_path'], 
                                            pair_obj['cue'],
                                            self.n_time, 
                                            self.cue_ratio, 
                                            -1)
        audio      = normalize(audio, norm_type=self.norm_type)
        return audio, cue
        
    def __len__(self):
        return len(self.pair_ids)