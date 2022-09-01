import torch
import torch.nn as nn

from torch.utils.data   import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def batchlize(dataset, batch_size, shuffle=False, custom=False):
    if custom:
        return DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          drop_last=True, 
                          collate_fn=batch_collect
                         )
    else:
        return DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          drop_last=True)
        

def batch_collect(batch):
    item_nums = len(batch[0])
    batch_out = ()
    for i in range(item_nums):
        if isinstance(batch[0][i], torch.Tensor):
            batch_item  = torch.stack(pad_sequence([item[i].permute(1, 0) for item in batch], 
                                                   batch_first=True).permute(0, 2, 1))
        else:
            batch_item  = torch.stack([item[i] for item in batch])
        batch_out  += (batch_item, )
    return batch_out
        

