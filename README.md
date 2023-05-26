# Semantically-informed Hierarchical Event Modeling

![Python Version](https://badgen.net/pypi/python/black)
![MIT License](https://img.shields.io/github/license/dipta007/SHEM?style=plastic)
![GitHub Top Language](https://img.shields.io/github/languages/top/dipta007/SHEM?style=plastic)

This repository is the official implementation of [Semantically-informed Hierarchical Event Modeling](https://arxiv.org/abs/2212.10547) (Published in *SEM 2023).

![Main Figure](./figs/main.png)

## Getting Started
```
git clone https://github.com/dipta007/SHEM
cd SHEM
git checkout ind_frame
```

## Conda Environment
```
conda create -n shem python=3.7
conda activate shem
conda install pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torchtext==0.2.3
pip install -U scikit-learn
conda install -c conda-forge pytablewriter
conda install -c anaconda pandas
pip install gdown
```

## Data:
```
conda activate shem
pip install gdown
mkdir saved_models
mkdir saved_configs
gdown https://drive.google.com/drive/u/1/folders/1YUfHGdyVXONtvbJQCbBN0OA0ibXP82bw -O ./data --folder
```


## Usage
### Changing Frame Relation:
There are in total 10 frame relations. To change the frame relation, change the value of the variable `frame_relation` in `./framenet_relations.py` to one of the following values:
```
frame_relation = 'Inheritance'
frame_relation = 'Using'
frame_relation = 'Precedes'
frame_relation = 'Metaphor'
frame_relation = 'See_also'
frame_relation = 'Causative_of'
frame_relation = 'Inchoative_of'
frame_relation = 'Perspective_on'
frame_relation = 'Subframe'
frame_relation = 'ReFraming_Mapping'
```
### Training:
```
./train.sh $obsv_prob $exp_num $seed
```

### Evaluation:

#### data_mode:
 {'valid','test'}

#### Perplexity:
```
./test_ppx.sh $obsv_prob $exp_num $seed $data_mode
```
#### Wiki Inverse Narrative Cloze:
```
./wiki_inv_narr.sh $obsv_prob $exp_num $seed $data_mode
```

## ⚠️ Disclaimer

Some parts of the code were inspired by [SSDVAE](https://github.com/mmrezaee/SSDVAE) implementations.