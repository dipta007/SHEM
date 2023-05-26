from collections import defaultdict

import numpy as np
from tqdm import tqdm

import wandb

ENTITY = "ssdvae-hierarchical"
PROJECT = "gaoetal_hssdvae"
METRIC_NAMES = ["val/hard", "val/hard_ext", "val/trans"]

api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}")

groups = set()
metrics = defaultdict(list)

for run in tqdm(runs):
    try:
        exp_name = run.name

        mx = defaultdict(int)
        for v in run.scan_history(keys=METRIC_NAMES):
            for metric in METRIC_NAMES:
                mx[metric] = max(mx[metric], v[metric])
        print(exp_name, mx)

        for metric in METRIC_NAMES:
            run.summary[f"max_{metric}"] = mx[metric]

        run.summary.update()
    except Exception as e:
        print(e)
        continue
