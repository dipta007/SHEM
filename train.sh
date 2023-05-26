#!/bin/bash
. /opt/anaconda/etc/profile.d/conda.sh
conda activate shem

python -u main.py\
    --obsv_prob "${1}"\
    --exp_num "${2}"\
    --seed "${3}"\
    --emb_size 300\
    --enc_hid_size 512\
    --dec_hid_size 512\
    --nlayers 2\
    --lr 0.001\
    --log_every 200\
    --save_after 500\
    --validate_after 2500\
    --clip 5.0\
    --epochs 40\
    --batch_size 64\
    --bidir 1\
    -max_decode_len 50\
    -num_latent_values 500\
    -latent_dim 500\
    -use_pretrained 1\
    -dropout 0.0\
    --num_clauses 5\
    --frame_max 500\
    -use_pretrained 1\
    --cuda \
