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
git checkout event_similarity
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
pip install hydra-core --upgrade
pip install hydra_colorlog --upgrade
pip install wandb
pip install prettytable
pip install transformers
conda install -c conda-forge spacy
conda install -c conda-forge cupy
python -m spacy download en_core_web_trf
pip install gdown
```

## Data:
```
conda activate shem
pip install gdown
gdown https://drive.google.com/drive/u/1/folders/1s16s33Fwgt1MNk7_YO7iJO32gW2TZvrv -O ./data --folder
```


## Usage
### Training:
```
python train.py +experiment=naacl ++exp_num=$exp_num ++seed=$SEED ++debug=False
```

### Evaluation:

### test_type:
 {'sim-hard', 'sim-ext', 'sim-ext'}

#### similarity evaluation:
```
python test.py --exp_name=naacl_$exp_num --test_type=$test_type
```