#!/bin/bash

#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sroydip1+ada@umbc.edu       # Where to send mail
#SBATCH -D .
#SBATCH --job-name="naa_66_1234"
#SBATCH --output=log/output/naa_66_1234.log
#SBATCH --error=log/error/naa_66_1234.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=240:00:00
#SBATCH --constraint=rtx_6000                   # NULL (12GB), rtx_6000 (24GB), rtx_8000 (48GB)

v=$(git status --porcelain | wc -l)
if [[ $v -gt 1000 ]]; then
    echo "Error: uncommited changes" >&2
    exit 1
else
    echo "Success: No uncommited changes"
    echo "CMD:" $@ ++debug=False
    $@ ++debug=False
fi
