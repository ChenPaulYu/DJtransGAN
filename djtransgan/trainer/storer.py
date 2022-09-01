import os
import torch
import torch.nn    as nn 
import torch.optim as optim
import matplotlib.pyplot as plt



from djtransgan.config  import settings
from djtransgan.utils   import check_exist, time_to_samples, purify_device
from djtransgan.utils   import out_audio, out_pt, out_json
from djtransgan.utils   import loss_visualize, save_figure
from djtransgan.dataset import DataLoaderSampler

class Storer():
    
    def __init__(self, 
                 dataset, 
                 n_sample, 
                 out_dir, 
                 hyper):
        
        
        sampler             = DataLoaderSampler(dataset, batch_size=n_sample, shuffle=False)
        stored_out          = sampler()
        
        self.out_dir        = out_dir
        self.pair_audio     = stored_out[0]
        self.cue            = stored_out[1]
        
        self.loss           = {'D': [], 'G': []}
        self.loss_type      = hyper['train']['loss_type']
        self.store_hyperparameter(hyper)
        

    # Store data
    
    def store_hyperparameter(self, hyper):
        out_path = os.path.join(self.out_dir, 'hyperparameter.json')
        check_exist(out_path)
        out_json(hyper, out_path)
        
    def store_net(self, net, epoch):
        out_dir    = os.path.join(self.out_dir, 'net', f'epoch_{epoch+1}')
        out_path = os.path.join(out_dir, 'generator.pth') 
        check_exist(out_path)
        out_pt(purify_device(net[0].state_dict()), out_path)
        
        out_path = os.path.join(out_dir, 'discriminator.pth') 
        check_exist(out_path)
        out_pt(purify_device(net[1].state_dict()), out_path)
                
    def store_loss(self, loss, epoch):
        
        out_dir   = os.path.join(self.out_dir, 'loss', f'epoch_{epoch+1}')
        
        fig       = loss_visualize(loss, plt, title=self.loss_type)
        
        out_path  = os.path.join(out_dir, f'loss_data.pt')
        check_exist(out_path)
        out_pt(loss, out_path)
        
        out_path  = os.path.join(out_dir, 'loss_figure.png')
        check_exist(out_path)
        save_figure(fig, out_path)
        
    def store_mix(self, mix_audio, mix_out, epoch):
        
        out_dir  = os.path.join(self.out_dir, 'mix', f'epoch_{epoch+1}')
        
        # save mix audio
        for i, audio in enumerate(mix_audio):
            out_path = os.path.join(out_dir, 'audio', f'{i}.wav')
            check_exist(out_path)
            out_audio(purify_device(audio), out_path)
            
        # save mix res
        out_path = os.path.join(out_dir, 'mix_out.pt')
        check_exist(out_path)
        out_pt(purify_device(mix_out), out_path)
    
                
    # Log
    def log_loss(self, epoch):
        print(f"epoch_{epoch+1} G={self.loss['G'][-1]:.4f},"
              f"epoch_{epoch+1} D={self.loss['D'][-1]:.4f}")
        

    def generate_mix(self, generator):
        generator.eval()
        device             = next(generator.parameters()).device        
        pair_audio         = [audio.to(device) for audio in self.pair_audio]
        cue                = self.cue.to(device)
        mix_audio, mix_out = generator.infer(*pair_audio, cue_region=cue)
        return (mix_audio, mix_out)
    
    def log(self, loss, epoch):
        self.loss['G'].append(loss[0])
        self.loss['D'].append(loss[1])
        self.log_loss(epoch)
        
    
    def __call__(self, net, epoch):
        
        # generate        
        if net is not None:
            
            # generate 
            mix_audio, mix_out = self.generate_mix(net[0])
            mix_out['cue']     = self.cue * settings.N_TIME
            
            # store            
            self.store_mix(mix_audio, mix_out, epoch)
            self.store_net(net, epoch)
            if epoch >= 0:
                self.store_loss(self.loss, epoch)
