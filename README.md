# DJTransGAN: Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks

> This repository contains the code for "[Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks](https://arxiv.org/abs/2110.06525)"
> *2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2022)*
> Bo-Yu Chen, Wei-Han Hsu, Wei-Hsiang Liao, Marco A. Martínez-Ramírez, Yuki Mitsufuji, Yi-Hsuan Yang

## Overview

The repo is under construction. We split the full implementation into five parts and plan to release it one by one in the future. These parts include 

1. Differentiable DJ mixer includes differentiable fader and differentiable equalizer in the time-frequency domain 
    - [x] differentiable fader
    - [x] differentiable equalizer
    - [x] differentiable mixer
2. DJTransGAN architecture and its training code 
    - [ ] GAN model architecture
    - [ ] GAN training code
3. DJTransGAN pre-trained model and its inference code
    - [ ] GAN pre-trained model
    - [ ] GAN inference code
4. Data generation pipeline includes four-step usually applied by DJ to select appropriate cue points and matched tracks. 
    - [ ] music structure segmentation
    - [ ] mixablity estimation and pairing
    - [ ] bpm and key alignment
    - [ ] cue point selection
5. Documentation

Furthermore, if you want to hear more audio example, please check our demo page [here](https://paulyuchen.com/djtransgan-icassp2022/).


## Dataset

We collected two datasets to train our proposed approaches: DJ mixset from [Livetracklist](https://www.livetracklist.com/) and individual EDM tracks from [MTG-Jamendo-Dataset](https://github.com/MTG/mtg-jamendo-dataset). 

To be more specific, We in-house collect long DJ mixsets from Livetracklist and only consider the mixset with mix tag to ensure the quality of the mixset. Furthermore, we select the individual EDM track from MTG-Jamendo-Dataset, which means the track with an EDM tag in the total collection. The detailed information about using these two datasets to train our model is described in Section 3.1; please check it if you are interested. 

Last, we can not provide our training dataset for reproducing our results because of the license issues. However, we will release our training code and pre-trained model in the future to make it more applicable. if you have any other suggestions, please let me know.

## Install

```

pip install -r requirements.txt

```

## Usage 

We release several usage examples in `examples` directories; we will keep updating them when a new implementation is released.

### Differentiable DJ mixer

You can choose to use fader and equalizer indivdually or use both in the same time. If you want to use it individually, please check `/examples/mixer/mask.ipynb`. If you want to use both in the same time, please check `/examples/mixer/mixer.ipynb`.


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

This repo is done during the internship in the Sony Group Corporation with outstanding mentoring by my incredible mentors in Sony [Wei-Hsiang Liao](https://jp.linkedin.com/in/wei-hsiang-liao-66283154), [Marco A. Martínez-Ramírez](https://m-marco.com/), and [Yuki Mitsufuji](https://www.yukimitsufuji.com/) and my colleague [Wei-Han Hsu](https://github.com/ddman1101) and advisor [Yi-Hsuan Yang](https://www.citi.sinica.edu.tw/pages/yang/) in Academia  Sinica. The results are a joint effort with Sony Group Corporation and  Academia  Sinica. I sincerely appreciate all the support made by them to make this research happen. Moreover, please check the other excellent AI research made by Sony [here](https://github.com/sony/ai-research-code) if you are interested.  



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
