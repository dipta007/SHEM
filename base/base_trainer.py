import json
import logging
import time
from collections import defaultdict

import torch
from tqdm import tqdm

from utils.utils import (
    ensure_dir,
    load_model,
    save_model,
    summary_model,
    to_device,
    wandb_alert,
    wandb_log,
    wandb_summary,
)
from wandb import AlertLevel

log = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, config, optimizer, model):
        self.config = config

        self.optimizer_partial = optimizer
        self.model = model
        self.model = to_device(self.model, self.config.device)
        self.optimizer = self.get_optimizer()
        self.schedulers = self.get_schedulers()

        summary_model(self.model)
        # wandb_watch(self.model)

        self.grad_accum = max(
            1, self.config.train.desired_batch_size // self.config.data.batch_size
        )
        log.info(f"Gradient accumulation: {self.grad_accum}")

        # will update later during running epoch
        self.epoch = 0
        self.total_iterations = 0
        self.log_dict = defaultdict(lambda: 0.0)

        self.es_patience_counter = 0
        self.es_best_metric = float("inf")
        if self.config.train.early_stopping.goal == "max":
            self.es_best_metric = float("-inf")

    def train(self):
        with tqdm(
            enumerate(self.train_loader), unit="batch", total=len(self.train_loader)
        ) as tepoch:
            for _, data in tepoch:
                tepoch.set_description(f"Epoch {self.epoch}")
                self.model.train()
                self.log_dict = defaultdict(lambda: 0.0)

                data = to_device(data, self.config.device)
                pred = self.model(data)
                loss = self.get_loss(pred, data, "train")
                # TODO: check if its needed or not
                # loss = loss / self.grad_accum

                loss.backward()
                self.total_iterations += 1

                # ? Gradient accumulation & Update
                if self.total_iterations % self.grad_accum == 0:
                    self.optimizer.step()
                    self.model.zero_grad()

                # ? Logging to wandb
                self.run_metrics(data, pred, "train")
                self.log(console_log=False)

                # ? Validation after an interval of steps
                if self.total_iterations % self.config.train.validate_every == 0:
                    self.eval()

                tepoch.set_postfix(loss=loss.item())

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        self.log_dict = defaultdict(lambda: 0.0)
        with tqdm(
            self.valid_loader, unit="batch", total=len(self.valid_loader)
        ) as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Validation")

                data = to_device(data, self.config.device)
                pred = self.model(data)
                loss = self.get_loss(pred, data, "val")

                # ? Logging to wandb
                self.run_metrics(data, pred, "val")

                tepoch.set_postfix(loss=loss.item())

        for key in self.log_dict.keys():
            self.log_dict[key] /= len(self.valid_loader)

        self.get_similarity("val")

        self.log(console_log=True)

        self.check_for_improvement()
        self.check_early_stopping()

    def fit(self, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        for epoch in range(self.config.train.epochs):
            start_time = time.time()

            self.epoch = epoch
            self.train()
            self.eval()

            for scheduler in self.schedulers:
                scheduler.step()

            log.info(f"Epoch {epoch} Completed. Took time: {time.time() - start_time}")
            start_time = time.time()

    def log(self, console_log=False):
        self.log_dict["iteration"] = self.total_iterations
        self.log_dict["epoch"] = self.epoch
        wandb_log(self.log_dict)
        if console_log:
            log.info(f"Current wandb dict:\n{json.dumps(self.log_dict, indent=4)}")

    def check_for_improvement(self):
        goal = self.config.train.early_stopping.goal
        metric = self.config.train.early_stopping.metric
        prev = self.es_best_metric
        if goal == "max":
            if self.log_dict[metric] > self.es_best_metric:
                self.es_best_metric = self.log_dict[metric]
                self.es_patience_counter = 0
            else:
                self.es_patience_counter += 1
        elif goal == "min":
            if self.log_dict[metric] < self.es_best_metric:
                self.es_best_metric = self.log_dict[metric]
                self.es_patience_counter = 0
            else:
                self.es_patience_counter += 1

        wandb_summary(f"{goal}_{metric}", self.es_best_metric)
        if self.es_patience_counter == 0:
            save_model(
                self.model, f"{self.config.train.model_dir}/{self.config.exp_name}.pt"
            )
            wandb_alert(
                title=f"{metric} Improved, goal: {goal}",
                text=f"Epoch: {self.epoch} Iteration: {self.total_iterations} Current {metric}: {self.log_dict[metric]} Previous {goal} {metric}: {prev}",
                level=AlertLevel.INFO,
            )
        else:
            wandb_alert(
                title=f"{metric} Not Improved, goal: {goal}",
                text=f"Epoch: {self.epoch} Iteration: {self.total_iterations} Patience: {self.es_patience_counter} Current {metric}: {self.log_dict[metric]} Current {goal} {metric}: {self.es_best_metric}",
                level=AlertLevel.WARN,
            )

    def check_early_stopping(self):
        metric = self.config.train.early_stopping.metric
        if self.config.train.early_stopping.enabled:
            if self.es_patience_counter >= self.config.train.early_stopping.patience:
                log.info(f"Early stopping at epoch {self.epoch}")
                msg = f"Early stopping at epoch {self.epoch} iteration {self.total_iterations}, Best {metric}: {self.es_best_metric}"
                raise RuntimeError(msg)

    def save_checkpoint(self):
        ensure_dir(self.config.train.checkpoint_dir)
        path = f"{self.config.train.checkpoint_dir}/checkpoint_epoch_{self.epoch}_iter_{self.total_iterations}.pt"
        torch.save(
            {
                "epoch": self.epoch,
                "total_iterations": self.total_iterations,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        log.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, epoch, total_iterations):
        path = f"{self.config.train.checkpoint_dir}/checkpoint_epoch_{epoch}_iter_{total_iterations}.pt"
        log.info(f"Loading checkpoint from path: {path}")
        checkpoint = torch.load(path)
        self.epoch = checkpoint["epoch"]
        self.total_iterations = checkpoint["total_iterations"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model = to_device(self.model)

    def run(self):
        log.info("Starting Training....")
        try:
            self.fit(self.train_dataloader, self.val_dataloader)
        except Exception as e:
            log.error(f"Error: {e}")

        self.test()

    def test(self):
        self.model = load_model(
            f"{self.config.train.model_dir}/{self.config.exp_name}.pt",
            self.config.device,
        )
        self.model.eval()
        log.info("No test provided/needed")

    def get_optimizer(self):
        log.info(f"Using optimizer: {self.optimizer_partial}")
        return self.optimizer_partial(params=self.model.parameters())

    def get_schedulers(self):
        return []

    def run_metrics(self, data, pred, mode="train"):
        log.info("No metrics provided/needed")

    def get_loss(self, pred, data, mode="train"):
        raise NotImplementedError
