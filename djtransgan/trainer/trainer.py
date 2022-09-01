import os
import time
import copy
import datetime
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


from djtransgan.config       import settings
from djtransgan.utils        import out_json, check_exist, get_device, purify_device, check_nan
from djtransgan.dataset      import batchlize, DataLoaderSampler
from djtransgan.trainer      import get_optimizer, get_criterion, Storer





class Trainer():
    def __init__(self, 
                 model_args, 
                 train_args, 
                 store_args):
        
        self.device          = get_device(train_args['n_gpu'])
        self.epochs          = train_args['epoch']        
        self.loss_type       = train_args['loss_type']
        self.n_critic        = train_args['n_critic']
        
        
        self.criterion       = get_criterion(train_args['loss_type'])
        self.generator       = model_args['G'].to(self.device)
        self.discriminator   = model_args['D'].to(self.device)
        
                
        dataset              = train_args['dataset']
        batch_size           = train_args['batch_size'] 
        
        self.datasampler     = DataLoaderSampler(dataset[0], batch_size=batch_size, shuffle=True) if len(dataset) > 1 else None
        self.dataloader      = batchlize(dataset[1], batch_size=batch_size, shuffle=True)
        
        self.otpim_g         = get_optimizer(self.generator, 
                                             train_args['optim'][0], 
                                             train_args['lr'][0])
        self.otpim_d         = get_optimizer(self.discriminator, 
                                             train_args['optim'][1], 
                                             train_args['lr'][1])
        
        self.log_interval    = store_args['log_interval']
        self.out_dir         = os.path.join(settings.STORE_DIR, store_args['out_dir'])
        self.begin_date      = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.hyper           = self.get_hyperparameter(model_args, train_args, store_args)
        self.storer          = Storer(dataset[0], 
                                      store_args['n_sample'], 
                                      self.out_dir, 
                                      self.hyper)
        
    
        
    def get_hyperparameter(self, 
                           model_args, 
                           train_args, 
                           store_args):
        
        train_args.pop('dataset')
        
        return {
            'model': {key: model.hyper for (key, model) in model_args.items()}, 
            'train': train_args, 
            'store': store_args, 
            'date' : self.begin_date
        }
    
    
    
    def average_loss(self, loss):
        return torch.mean(torch.tensor(loss)).item()
                          
    def generate_mix(self):
        pair_audios, cue = self.datasampler()
        cue         = cue.to(self.device)
        pair_audios = [audio.to(self.device) for audio in pair_audios]
        mix_mag, mix_out = self.generator(*pair_audios, cue_region=cue)
        mix_mag = torch.sum(mix_mag, axis=1) if len(mix_mag.size()) == 5 else mix_mag
        return (mix_mag, mix_out)
    
        
    def train_step_G(self):    
        self.generator.zero_grad()
        fake_mix, fake_mix_out = self.generate_mix()
        dgz  = self.discriminator(fake_mix)
        loss = self.criterion(dgz)
        loss.backward()
        self.otpim_g.step()        
        return loss.item()
        

        
    def train_step_D(self, real_mix, real_cue):
        self.discriminator.zero_grad()
        with torch.no_grad():
            fake_mix, fake_mix_out = self.generate_mix()
        
        dgz  = self.discriminator(fake_mix)
        dx   = self.discriminator(real_mix)
        loss = self.criterion(dgz, dx) 
        loss.backward()                
        self.otpim_d.step()
        return loss.item()
                          
        
        
        
    def train(self):
        
        begin_time = time.time()
        D_Losses   = []
        
        self.storer([self.generator, self.discriminator], -1)
        for epoch in range(self.epochs):            
            for batch_idx, (real_mix, real_cue) in enumerate(tqdm(self.dataloader)):
                
                real_mix = real_mix.to(self.device)
                real_cue = real_cue.to(self.device)
                
                D_Losses.append(self.train_step_D(real_mix, real_cue)) # train D for one step
                if check_nan(D_Losses[-1]):
                    print(f'Batch: {batch_idx+1}, Epoch: {epoch+1}, D Loss is nan .....')
            
                
                if (batch_idx+1) % self.n_critic == 0:
                    G_Loss = self.train_step_G()
                    D_Loss = self.average_loss(D_Losses)
                    D_Losses.clear()
                    
                    if check_nan(G_Loss): 
                        print(f'Batch: {batch_idx+1}, Epoch: {epoch+1}, G Loss is nan .....')
                        break
                    
                    if (batch_idx+1) % self.log_interval == 0:
                        self.storer.log([G_Loss, D_Loss], epoch)
                        
            self.storer([self.generator, self.discriminator], epoch)
            print(f'[{epoch + 1}/{self.epochs}] one epoch completed ...')
            
            
        print('Train finished. Elapsed: %s' % datetime.timedelta(seconds=time.time() - begin_time))
            