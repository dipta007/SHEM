import json
import linecache
import logging
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from prettytable import PrettyTable
from torch import nn

import wandb
from wandb import AlertLevel

log = logging.getLogger(__name__)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError
        return self.get(attr, None)

    __delattr__ = dict.__delitem__
    __setattr__ = dict.__setitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = DotDict(value)
            self[key] = value


def seed_everything(seed=42):
    log.info("Seeding with {}....".format(seed))
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # if input size is fixed for every iteration, set to True
    torch.backends.cudnn.benchmark = True


def get_repo_name():
    repo_name = os.path.basename(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .strip()
        .decode()
    )
    return repo_name


def get_commit_hash():
    return subprocess.check_output(["git", "describe", "--always"]).strip().decode()


def print_repo_info():
    log.info("*" * 44)
    log.info(" ".join(sys.argv))
    log.info("")
    git_commit_hash = get_commit_hash()
    repo_name = get_repo_name()
    log.info(f"Git repo: {repo_name}")
    log.info(f"Git commit hash: {git_commit_hash}")
    log.info("*" * 44)


def print_config(config):
    log.info("-" * 44)
    log.info(f"CMD: {' '.join(sys.argv)}")
    log.info("")
    log.info(f"Hydra Configs:\n{json.dumps(config, indent=4)}")
    log.info("-" * 44)


def to_device(data, device):
    device = torch.device(device)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        return [to_device(x, device) for x in data]
    if isinstance(data, dict):
        return DotDict({k: to_device(v, device) for k, v in data.items()})
    if isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)
    if isinstance(data, nn.Module):
        return data.to(device)
    if isinstance(data, nn.ModuleList):
        return nn.ModuleList([to_device(x, device) for x in data])
    if isinstance(data, nn.Parameter):
        return data.to(device)
    if isinstance(data, nn.ParameterList):
        return nn.ParameterList([to_device(x, device) for x in data])
    if isinstance(data, nn.Embedding):
        return data.to(device)

    return data.to(device)
    raise TypeError(f"Unsupported type: {type(data)}")


def wandb_alert(title, text="", level=AlertLevel.INFO):
    try:
        wandb.alert(get_repo_name() + " - " + title, text, level)
    except Exception as e:
        pass


def wandb_log(wandb_dict):
    try:
        wandb.log(wandb_dict)
    except Exception as e:
        pass


def wandb_summary(name, value, update=None):
    try:
        if update is None or name not in wandb.run.summary.keys():
            wandb.run.summary[name] = value
        else:
            wandb.run.summary[name] = update(wandb.run.summary[name], value)
    except Exception as e:
        pass


def wandb_watch(model):
    try:
        wandb.watch(model, log="all")
    except Exception as e:
        pass


def wandb_log_code(run):
    def include(path):
        included_files = [
            ".py",
            ".yaml",
            ".ipynb",
            ".md",
            ".sh",
        ]
        if any(path.endswith(ext) for ext in included_files):
            return True
        return False

    def exclude(path):
        exclude_dirs = [
            "__pycache__",
            "wandb",
            ".history",
            ".git",
            ".idea",
            ".vscode",
            ".venv",
            ".DS_Store",
            ".ipynb_checkpoints",
            ".vector_cache",
            "nltk_data",
            "corpora" "log",
            "experiments",
        ]
        if any(f"{ext}/" in path for ext in exclude_dirs):
            return True
        return False

    run.log_code("./", include_fn=include, exclude_fn=exclude)


def save_model(model, path):
    dir = "/".join(path.split("/")[:-1])
    ensure_dir(dir)
    torch.save(model, path)


def load_model(path, device="cpu"):
    return torch.load(path, map_location=device)


def save_pickle(obj, path):
    dir = "/".join(path.split("/")[:-1])
    ensure_dir(dir)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def summary_model(model, show_all=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and not show_all:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    log.info(f"\n{table}\n")
    count_parameters(model)
    return total_params


def count_parameters(model):
    log.info(f"Number of All Parameters: {sum(p.numel() for p in model.parameters())}")
    log.info(
        f"Number of Required grad Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )


# TODO: need to add more cases
def is_equal(a, b):
    if isinstance(a, tuple):
        for i in range(len(a)):
            if not is_equal(a[i], b[i]):
                return False
            return True
    if isinstance(a, dict):
        for k, v in a.items():
            if not is_equal(v, b[k]):
                print(k)
                return False
        return True
    if isinstance(a, list):
        for i, v in enumerate(a):
            if not is_equal(v, b[i]):
                return False
        return True
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)

    return a == b


def get_file_line_count(p):
    tot = subprocess.check_output(["wc", "-l", p]).decode().strip().split()[0]
    return int(tot)


def get_file_line(file, line_num):
    # file is indexed from 1
    line_num = line_num + 1
    return linecache.getline(file, line_num).strip()


def safe_log(val):
    eps = 1e-7
    return torch.log(val + eps)
