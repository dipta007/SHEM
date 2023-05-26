#!/bin/bash
. /opt/anaconda/etc/profile.d/conda.sh
conda activate shem

CUDA_VISIBLE_DEVICES=0\
    python inference.py\
    --obsv_prob "${1}"\
    --exp_num "${2}"\
    --seed "${3}"\
    --data_mode "${4}"\
    --cuda \

