import os
import gdown
from djtransgan.utils import check_exist


def get_url(g_id):
    return f'https://drive.google.com/u/0/uc?id={g_id}&export=download'


def download_pretrained():    
    # download minmax djtransgan
    net_url    = get_url('1L1TXWY7YOJEaozrFr4MqOZ5aXwJ0_o31')
    out_path   = os.path.join('./pretrained/djtransgan_minmax.pt')
    check_exist(out_path)
    if os.path.exists(out_path):
        print(f'{out_path} exist ...')
    else:
        gdown.download(net_url, out_path)
        print(f'{out_path} download ...')
    
    # download least square djtransgan
    net_url    = get_url('1JtBUJL3sERl5HaM7sSH7p2Tw5Wnejvtt')
    out_path   = os.path.join('./pretrained/djtransgan_least_square.pt')
    check_exist(out_path)
    
    if os.path.exists(out_path):
        print(f'{out_path} exist ...')
    else:
        gdown.download(net_url, out_path)
        print(f'{out_path} download ...')