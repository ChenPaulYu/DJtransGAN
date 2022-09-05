# DJtransGAN: Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks

> This repository contains the code for "[Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks](https://arxiv.org/abs/2110.06525)"
> *2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2022)*
> Bo-Yu Chen, Wei-Han Hsu, Wei-Hsiang Liao, Marco A. Martínez-Ramírez, Yuki Mitsufuji, Yi-Hsuan Yang

## Overview


The reop is nearly complete; we have already open source all the code you need for not only use (provide a pre-trained model) but also train the DJtransGAN. We will keep improving the document and add more visualization soon. Currently, the repo contains 3/4 of the DJtransGAN code include

1. Differentiable DJ mixer includes differentiable fader and differentiable equalizer in the time-frequency domain 
2. DJtransGAN architecture and its training code 
3. DJtransGAN pre-trained model and its inference code

The remaining 1/4 of DJtransGANs' is implemented in another repo [DJtransGAN-dg-pipeline](https://github.com/ChenPaulYu/DJtransGAN-dg-pipeline) to make the codebase clean. Moreover, These two repos are totally independent and can be executed individually, and we will detail the process below.


Furthermore, if you want to hear more audio example, please check our demo page [here](https://paulyuchen.com/djtransgan-icassp2022/).


## Dataset

We collected two datasets to train our proposed approaches: DJ mixset from [Livetracklist](https://www.livetracklist.com/) and individual EDM tracks from [MTG-Jamendo-Dataset](https://github.com/MTG/mtg-jamendo-dataset). 

To be more specific, We in-house collect long DJ mixsets from Livetracklist and only consider the mixset with **mix** tag to ensure the quality of the mixset. Furthermore, we select the individual EDM track from MTG-Jamendo-Dataset, which means the track with an **EDM** tag in the total collection. The detailed information about using these two datasets is described in Section 3.1; please check it if you are interested in. 

Unfortunately, we can not provide our training dataset for reproducing the results because of license issues. However, we release the training code and pre-trained model for you to try. Contact me or open the pull request if you have any other issues. 


## Setup 

### Install 
```

pip install -r requirements.txt

```

### Generate dataset
You should first clone the [DJtransGAN-dg-pipeline](https://github.com/ChenPaulYu/DJtransGAN-dg-pipeline) and refer to its README.md to generate dataset include mixable  pairs and mixes made by professional DJ. 

```

git clone https://github.com/ChenPaulYu/DJtransGAN-dg-pipeline

```

### Configuration
Next, you should set the configuration in `djtransgan/config/settings.py`  for global usage of the repo, Most important of all, you should set the path of `PAIR_DIR`, `MIX_DIR` and  `STORE_DIR`.

1. `PAIR_DIR` : the directory contain a collection of mixable pair and its cue points which is generate by [DJtransGAN-dg-pipeline](https://github.com/ChenPaulYu/DJtransGAN-dg-pipeline).
2. `MIX_DIR` : the directory contain a collection of mix segment and its cue points which is generate by [DJtransGAN-dg-pipeline](https://github.com/ChenPaulYu/DJtransGAN-dg-pipeline).
3. `STORE_DIR` : the directory conatian the training and inferring result of DJtransGAN.



## Usage 

We release several examples in `examples/` and `script/` for  not only training and infering but also individual usage of each component (e.g: differentiable fader, equalizer and mixer).  Detail describe below. 

### Differentiable DJ mixer
You can choose to use fader and equalizer indivdually or use both in the same time. If you want to use it individually, please check `examples/mixer/mask.ipynb`. If you want to use both in the same time, please check `examples/mixer/mixer.ipynb`.

### Training

To train a DJtransGAN, you  need to run the script in `script/train.py`

```
  
python script/train.py [--lr=(list, ex: [1e-5, 1e-5])] [--log=(int, ex: 20)] [--n_gpu=(int, ex: 0)] [--epoch=(int, ex: 20)] [--out_dir=(str, ex: 'gan')] [--n_sample=(int, ex: 4)] [--n_critic=(int, ex: 1)] [--cnn_type=(str, e.g: res_cnn, cnn)] [--loss_type=(str, e.g: minmax, least_square)] [--bath_size=(int, ex: 4)]

```

- `--lr` :  learning rate of generator and discriminator (should provide two value). 
- `--log` :  log interval during the GAN training. 
- `--n_gpu` : whcih gpu you want to use 
- `--epoch` :  number of epoch which indicate how many time of dataset you want to train. 
-  `--out_dir` : the output directory which is going to save the training result. 
- `--n_sample` : the number of sample (mix) the model will generate in the end of every epoch. 
- `--n_critic` :   how many time the discriminator training over generator training. 
-  `--cnn_type` :  the cnn type of encoder (e.g: `res_cnn` or `cnn`) .  
- `--loss_type` :  the loss function during the GAN training support `minmax`  and `least square` loss. 
- `--batch_size` :  the batch size of dataloader during the GAN training


### Inference 
To generate the mix by trained generator, you need to run the script in `script/inference.py`. We provide two tracks in `test/` for your reference.

```
  
python script/inference.py [--g_path=(str, ex:'./pretrained/djtransgan_minmax.pt')] [--out_dir=(str, ex: 'results/inference')] [--prev_track=(str, ex: './test/Breikthru ft Danny Devinci-Touch.mp3')] [--next_track=(str, ex: './test/Jameson-Hangin.mp3')] [--prev_cue=(float, ex:96)] [--next_cue=(float, ex:30)] [--download=(bool, ex:1)]


```

- `--g_path` :  the path of trained generator, can be the pre-trained model provide by us or the model training by you.  
- `--out_dir` : the output directory which is going to save the result. 
- `--prev_track` : the path of the previous track (first track). 
- `--next_track` : the path of the next track (second track).  
-  `--prev_cue` :  the cue point of previous track (the point previous track totally fade out).  
- `--next_cue` :   the cue point of next track (the point next track totally fade in).  
- `--download` :   specify whether download the pre-trained model provided by us. 





## Citation

If you use any of our code in your work please consider citing us.

```
  @inproceedings{chen2022djtransgan,
    title={Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks},
    author={Chen, B. Y., Hsu, W. H., Liao, W. H., Ramírez, M. A. M., Mitsufuji, Y., & Yang, Y. H.},
    booktitle={ICASSP},
    year={2022}}
```

## Acknowledgement

This repo is done during the internship in the Sony Group Corporation with outstanding mentoring by my incredible mentors in Sony [Wei-Hsiang Liao](https://jp.linkedin.com/in/wei-hsiang-liao-66283154), [Marco A. Martínez-Ramírez](https://m-marco.com/), and [Yuki Mitsufuji](https://www.yukimitsufuji.com/) and my colleague [Wei-Han Hsu](https://github.com/ddman1101) and advisor [Yi-Hsuan Yang](https://www.citi.sinica.edu.tw/pages/yang/) in Academia  Sinica. The results are a joint effort with Sony Group Corporation and  Academia  Sinica. I sincerely appreciate all the support made by them to make this research happen. Moreover, please check the other excellent AI research made by Sony [here](https://github.com/sony/ai-research-code) and their recent work ["FxNorm-automix"](https://marco-martinez-sony.github.io/FxNorm-automix/) and ["distortionremoval"](https://joimort.github.io/distortionremoval/) which is going to present in ISMIR 2022.



## License
Copyright © 2022 Bo-Yu Chen

Licensed under the MIT License (the "License"). You may not use this
package except in compliance with the License. You may obtain a copy of the
License at

    https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
