import warnings
warnings.filterwarnings('ignore')

import os
import sys
if os.getcwd() in sys.path:
    sys.path.append('../')
else:    
    sys.path.append(os.getcwd())
    
import torch
import argparse

from djtransgan.config  import settings
from djtransgan.utils   import download_pretrained
from djtransgan.utils   import check_exist, time_to_str, squeeze_dim, get_filename
from djtransgan.utils   import load_pt, load_audio, out_audio
from djtransgan.model   import get_generator
from djtransgan.dataset import get_dataset, batchlize
from djtransgan.process import preprocess, postprocess



torch.manual_seed(settings.RANDOM_SEED)




def main():
    
    parser  = argparse.ArgumentParser(description='GAN Trainer')
    
    parser.add_argument('--out_dir'   , type=str, default=os.path.join(settings.STORE_DIR, 'inference'))
    parser.add_argument('--g_path'    , type=str, default='./pretrained/djtransgan_minmax.pt')
    parser.add_argument('--prev_track', type=str, default='./test/Breikthru ft Danny Devinci-Touch.mp3')
    parser.add_argument('--next_track', type=str, default='./test/Jameson-Hangin.mp3')
    parser.add_argument('--prev_cue'  , default = 96)
    parser.add_argument('--next_cue'  , default = 30)
    parser.add_argument('--download'  , default = 1)
    
     
    args       = parser.parse_args()
    
    if args.download:
        print('Download pre trained start ...')
        download_pretrained()
        print('Download pre trained complete ...')
        
    # Load generator
    print('Loading generator start ...')
    generator = get_generator()
    
    if os.path.exists(args.g_path):
        generator.load_state_dict(load_pt(args.g_path))
    else:
        print(f'{args.g_path} not exist')
    generator.eval()
    print('Loading generator complete ...')
    
    # Load audio
    print('Loading audio start ...')
    prev_audio = load_audio(args.prev_track)
    next_audio = load_audio(args.next_track)
    prev_cue   = args.prev_cue
    next_cue   = args.next_cue
    print('Loading audio complete ...')
    
    # Mix
    print('Mixing audio start ...')
    (pair_audio, timestamps), (pair_audio_for_g, cue_for_g) = preprocess(prev_audio, next_audio, prev_cue, next_cue)
    mix_audio, mix_out       = generator.infer(*pair_audio_for_g, cue_region=cue_for_g)
    post_mix_audio, post_cue = postprocess(mix_audio, pair_audio, timestamps, cue_for_g)
    saved_id                 = f'{get_filename(args.prev_track)}_{get_filename(args.next_track)}'
    print('Mixing audio complete ...')
    
    # save out 
    print('Saving audio start ...')
    out_path = os.path.join(args.out_dir, f'{saved_id}_short.wav')
    check_exist(out_path)
    out_audio(squeeze_dim(mix_audio).to(torch.float32), out_path)
    out_path = os.path.join(args.out_dir, f'{saved_id}_full.wav')
    check_exist(out_path)
    out_audio(squeeze_dim(post_mix_audio).to(torch.float32), out_path)
    print('Saving audio complete ...')



if __name__ == "__main__":
    main()