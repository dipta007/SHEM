#!/bin/bash
. /opt/anaconda/etc/profile.d/conda.sh
source activate shem

python wiki_val_generate.py\
    --obsv_prob "${1}"\
    --exp_num "${2}"\
    --seed "${3}"\
    --data_mode "${4}"\
    --cuda \
