import torch
import random
from acoustics.generator import noise
from djtransgan.config   import settings


random.seed(settings.RANDOM_SEED)


def generate_noise(secs, color='white'):
    return torch.from_numpy(noise(secs * settings.SR, color)).to(torch.float32)