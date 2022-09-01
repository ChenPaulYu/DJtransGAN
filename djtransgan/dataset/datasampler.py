import torch
import torch.nn as nn
from torch.utils.data   import Dataset, DataLoader
from djtransgan.dataset import batchlize

class DataLoaderSampler():
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True):
        self.count      = 0
        self.dataset    = dataset
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.shuffle    = shuffle 
        
        self.current    = self.get_new_dataloader()
        self.length     = len(self.current) 
        
        
    def get_new_dataloader(self):        
        return iter(batchlize(self.dataset, self.batch_size, shuffle=self.shuffle))
        
    def __call__(self):
        self.count      += 1
        if self.count > self.length:
            self.current = self.get_new_dataloader()
            self.length  = len(self.current) 
            self.count  += 1

        return next(self.current)