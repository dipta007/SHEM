import argparse
import logging
import os

from inference.similarity import Similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

from utils.utils import (
    DotDict,
    load_model,
    print_config,
    print_repo_info,
    seed_everything,
    to_device,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(config):
    config = DotDict(config)

    config.cuda = config.cuda and (
        torch.cuda.is_available() or torch.backends.mps.is_available()
    )
    config.device = (
        ("cuda" if torch.cuda.is_available() else "mps") if config.cuda else "cpu"
    )

    model = load_model(
        f"./experiments/{config.exp_name}/models/{config.exp_name}.pt",
        config.device,
    )
    model = to_device(model, config.device)

    try:
        config_from_model = model.config
        config_from_model.update(config)
        config = DotDict(config_from_model)
    except Exception as e:
        print(e)
        pass

    print_repo_info()
    print_config(config)

    log.info(f"Running on {config.device}")
    seed_everything(config.seed)

    # ? Different Tests
    if config.test_type == "sim-hard":
        Similarity(config, model, f"./data/inference/hard.txt").test()
    elif config.test_type == "sim-ext":
        Similarity(config, model, f"./data/inference/hard_extend.txt").test()
    elif config.test_type == "sim-trans":
        Similarity(
            config, model, f"./data/inference/transitive.txt", hard=False
        ).test()
    else:
        raise Exception("Unknown test type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="exp_name")
    parser.add_argument("--test_type", type=str, required=True, help="test_type")
    parser.add_argument("--cuda", type=bool, default=True, help="cuda")
    parser.add_argument(
        "--max_decode",
        type=int,
        default=2000,
        help="maximum decode for inverse narrative",
    )
    args = parser.parse_args()
    args = vars(args)
    main(args)
